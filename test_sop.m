clc; clear; close all;

addpath(genpath('externals/MBS-master'));
addpath(genpath('externals/utilities'));

path_video = '/home/zhuo/work/dataset/CVPR2016_VOS_Benchmark/davis/JPEGImages/480p';
path_flow = '/home/zhuo/work/dataset/CVPR2016_VOS_Benchmark/davis/OpticalFlow/siftFlow/backward';
path_objmask = 'results/objmask';

dir_save = 'results/som';

items = dir(path_video);
video_list = {};
for k = 1:numel(items)
    name = items(k).name;
    if ~isdir([path_video name]) && ~any(strcmp(name, {'.', '..'}))
        video_list{end+1} = name;  %#ok
    end
end

factor = 0.85;
nframes = 5;

for k = 1:numel(video_list)
    video = video_list{k};
    dir_save_name = fullfile(dir_save, video);
    if exist(dir_save_name, 'dir')==0
        mkdir(dir_save_name);
    end        
    
    frame_list = dir(fullfile(path_video, video, '*.jpg'));

    mask_list = cell(0);    
    flow_list = cell(0);
    for frame_id = 2:numel(frame_list)
        
        %% load optical flow        
        flow = readFlowFile(fullfile(path_flow, video, strcat(num2str(frame_id-1,'%05d'),'.flo')));
        [h, w, ~] = size(flow);
        
        %% MB saliency        
        data = cat(3, flow, zeros(h,w));
        prob_sal = saliency(data);
        
        %% segment first frame
        if frame_id == 2           
            objness_mask = imread(fullfile(path_objmask, video, strcat(num2str(frame_id-2,'%05d'),'.jpg')));                     

            sal_mask = segment_motion(-flow);            
            result = sal_mask & objness_mask;
            
            mask_list{1,1} = result;
            flow_list{1,1} = flow;
            
            save_name = fullfile(dir_save, video, strcat(num2str(frame_id-2,'%05d'),'.png'));
            imwrite(result, save_name);
        end  
        
        %% load objectness        
        seg_objness = imread(fullfile(path_objmask, video, strcat(num2str(frame_id-1,'%05d'),'.jpg')));         
        seg_objness = seg_objness>100;
        
        %% multi-frame accumulation of the foreground
        propagate_mask = accumulateMasks(mask_list, flow_list);   
        
        prob_fuse_sal = fuseProb(prob_sal, propagate_mask, factor);        
        multi_seg_sal = adaptiveSegment(prob_fuse_sal, 2);   
        
        %% multi-frame accumulation of the objectness

        prob_fuse_objness = fuseProb(seg_objness, propagate_mask, factor);        
        multi_seg_objness = adaptiveSegment(prob_fuse_objness, 1);           
  
        %% final segmentation
        multi_seg_sal = imdilate(multi_seg_sal, strel('disk', 6)); 
        result = multi_seg_sal & multi_seg_objness;        
        
        %% save history information  
        len = 1+numel(flow_list);
        mask_list{len,1} = result;
        flow_list{len,1} = flow;
        if numel(flow_list)>nframes
            mask_list(1) = [];
            flow_list(1) = [];
        end
        
        save_name = fullfile(dir_save, video, strcat(num2str(frame_id-1,'%05d'),'.png'));
        imwrite(result, save_name);
        
        fprintf('som, current processing video %d/%d, frame %d/%d \n', ...
            k, numel(video_list), frame_id, numel(frame_list)); 

    end    
    
end


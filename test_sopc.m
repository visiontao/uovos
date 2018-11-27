clc; clear; close all;

addpath(genpath('externals/MBS-master'));
addpath(genpath('externals/densecrf'));
addpath(genpath('externals/utilities'));

path_video = '/home/zhuo/work/dataset/CVPR2016_VOS_Benchmark/davis/JPEGImages/480p';
path_sop = 'results/sop';

dir_save = 'results/sopc';

items = dir(path_video);
video_list = {};
for k = 1:numel(items)
    name = items(k).name;
    if ~isdir([path_video name]) && ~any(strcmp(name, {'.', '..'}))
        video_list{end+1} = name;  %#ok
    end
end

for k = 1:numel(video_list)
    video = video_list{k};
    dir_save_name = fullfile(dir_save, video);
    if exist(dir_save_name, 'dir')==0
        mkdir(dir_save_name);
    end        
    
    frame_list = dir(fullfile(path_video, video, '*.jpg'));
    
    for frame_id = 1:numel(frame_list)
        
        %% load optical flow        
        frame = imread(fullfile(path_video, video, strcat(num2str(frame_id-1,'%05d'),'.jpg')));
        init_label = imread(fullfile(path_sop, video, strcat(num2str(frame_id-1,'%05d'),'.png')));
        
        init_label = imdilate(init_label, strel('disk', 3)); 
        
        %% apply crf refinement
        result = apply_crf(frame, init_label);
        
        result = result > 0;
        save_name = fullfile(dir_save, video, strcat(num2str(frame_id-1,'%05d'),'.png'));
        imwrite(result, save_name);
        
        fprintf('crf, current processing video %d/%d, frame %d/%d \n', ...
            k, numel(video_list), frame_id, numel(frame_list)); 

    end    
    
end


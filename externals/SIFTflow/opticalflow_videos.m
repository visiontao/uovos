close all; clear;clc

addpath(genpath('.'));


path_base = '~/data/CVPR2016_VOS_Benchmark/davis/';


%% get all videos from base folder
path_folder = '~/CVPR2016_VOS_Benchmark/davis/JPEGImages/480p/';

%list all sub-folders
contents = dir(path_folder);
videos = {};
for k = 1:numel(contents),
    name = contents(k).name;
    if isdir([path_folder name]) && ~any(strcmp(name, {'.', '..'})),
        videos{end+1} = name;  %#ok
    end
end


for n = 1:numel(videos)
    video = videos{n};

    path_img = strcat(path_base, 'JPEGImages/480p/', video);
    path_save = strcat(path_base, 'OpticalFlow/backward/', video);
    dir_img = dir(strcat(path_img,'/','*.jpg'));

    if exist(path_save, 'dir')==0
        mkdir(path_save);
    end

    
    for k = 2:numel(dir_img)

        tic;         
        %% load image frame
        frame2 = imread(strcat(path_img,'/', dir_img(k).name));    
        frame1 = imread(strcat(path_img,'/', dir_img(k-1).name));    

        flow = opticalFlow_siftFlow(frame2,frame1);

        saveName = strcat(path_save,'/flow_',num2str(k,'%05d'),'.mat');

        save(saveName, 'flow');

        duration = toc;

        fprintf('current processing video %d/%d, frame %d/%d, elapsed time %fs \n', ...
            n, numel(videos), k, numel(dir_img), duration); 

    end

end

close all; clear;clc

addpath(genpath('.'));

%% image path
video = 'goat';

path_base = '~/data/CVPR2016_VOS_Benchmark/davis/';
path_imgs = strcat(path_base, 'JPEGImages/480p/', video);
% path_save = strcat(path_base, 'OpticalFlow/480p/', video);
path_save = strcat(path_base, 'OpticalFlow/backward/', video);
dir_img = dir(strcat(path_imgs,'/','*.jpg'));

if exist(path_save, 'dir')==0
    mkdir(path_save);
end

%% load image frame
frame1 = imread(strcat(path_imgs,'/', dir_img(1).name));    

for k = 2:numel(dir_img)
    
    tic; 
    frame2 = imread(strcat(path_imgs,'/', dir_img(k).name));    
    
    flow = opticalFlow_siftFlow(frame2,frame1);
        
    saveName = strcat(path_save,'/flow_',num2str(k,'%05d'),'.mat');
    
    save(saveName, 'flow');
    
    duration = toc;
    
    fprintf('current processing frame %d/%d, elapsed time %fs \n', k, numel(dir_img), duration); 
    
    frame1 = frame2;
end

clc
close all
clear all

% Load and preprocess the leaf image
[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
I = imread([pathname, filename]);
I = imresize(I, [256, 256]);

% Enhance Contrast
I = imadjust(I, stretchlim(I));
figure, imshow(I);
title('Contrast Enhanced');

% Otsu Segmentation
I_Otsu = im2bw(I, graythresh(I));

%% Extract Features

% Feature Extraction
x = inputdlg('Enter the cluster no. containing the ROI only:');
cluster_number = str2double(x);
seg_img = I_Otsu;

if ndims(seg_img) == 3
    img = rgb2gray(seg_img);
end


% Create the Gray Level Co-occurrence Matrices (GLCMs)
glcms = graycomatrix(img, 'NumLevels', 8, 'GrayLimits', [], 'Offset', [0 1; -1 1; -1 0; -1 -1]);


% Derive Statistics from GLCM
stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1 - (1 / (1 + a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
in_diff = 0;

for i = 1:size(seg_img, 1)
    for j = 1:size(seg_img, 2)
        temp = seg_img(i, j) / (1 + (i - j)^2);
        in_diff = in_diff + temp;
    end
end
IDM = double(in_diff);

feat_disease = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, ...
    Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

% Load All The Features
load('Training_Data.mat');

% Define disease labels (0: Downy Mildew, 1: Powdery Mildew, 2: Black Rot, 3: Anthracnose)
disease_labels = [0; 1; 2; 3];
Train_Label = repmat(disease_labels, size(Train_Feat, 1) / 4, 1);

% Put the test features into variable 'test'
test = feat_disease;
result = multisvm(Train_Feat, Train_Label, test);

% Visualize Results
if result == 0
    helpdlg('Downy Mildew');
    disp('Downy Mildew');
elseif result == 1
    helpdlg('Powdery Mildew');
    disp('Powdery Mildew');
elseif result == 2
    helpdlg('Black Rot');
    disp('Black Rot');
elseif result == 3
    helpdlg('Anthracnose');
    disp('Anthracnose');
end

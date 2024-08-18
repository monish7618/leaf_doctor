function autoClusterGUI()

    % Create GUI figure
    fig = figure('Name', 'Auto Cluster Selection GUI', 'NumberTitle', 'off', 'Position', [100, 100, 400, 300]);

    % Create UI components
    btnLoadImage = uicontrol('Style', 'pushbutton', 'String', 'Load Image', 'Position', [20, 250, 100, 30], 'Callback', @loadImage);
    btnProcessImage = uicontrol('Style', 'pushbutton', 'String', 'Process Image', 'Position', [150, 250, 100, 30], 'Callback', @processImage);
    axImage = axes('Parent', fig, 'Position', [0.1, 0.1, 0.8, 0.6]);

    % Initialize variables
    loadedImage = [];
    segmentedImage = [];

    function loadImage(~, ~)
        % Load and display image
        [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
        if ~isequal(filename, 0)
            loadedImage = imread(fullfile(pathname, filename));
            loadedImage = imresize(loadedImage, [256, 256]);
            imshow(loadedImage, 'Parent', axImage);
            title(axImage, 'Original Image');
        end
    end

    function processImage(~, ~)
        if isempty(loadedImage)
            msgbox('Please load an image first.', 'Error', 'error');
            return;
        end

        % Enhance Contrast
        enhancedImage = imadjust(loadedImage, stretchlim(loadedImage));

        % Otsu Segmentation
        binaryImage = im2bw(enhancedImage, graythresh(enhancedImage));

        % Feature Extraction
        features = extractFeatures(binaryImage);

        % Automatic Cluster Selection
        [clusteredFeatures, result] = autoClusterSelection(features);

        % Display the segmented image
        segmentedImage = bsxfun(@times, loadedImage, cast(clusteredFeatures, 'like', loadedImage));
        figure, imshow(segmentedImage);
        title('Segmented ROI');

        % Display the result
        displayResult(result);
    end

    function features = extractFeatures(seg_img)
        if ndims(seg_img) == 3
            img = rgb2gray(seg_img);
        else
            img = seg_img;
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

        features = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, ...
            Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
    end

    function [clusteredFeatures, result] = autoClusterSelection(testFeatures)
        % Load All The Features
        load('Training_Data.mat');

        % Use k-means to automatically determine the optimal number of clusters
        maxClusters = min(10, size(Train_Feat, 1));
        [~, clusterCenters] = kmeans(Train_Feat(:, 1:13), maxClusters, 'distance', 'sqEuclidean', 'Replicates', 3);

        % Assign each training sample to the nearest cluster
        [~, clusterIdx] = pdist2(clusterCenters, Train_Feat(:, 1:13), 'squaredeuclidean', 'Smallest', 1);

        % Find the cluster with the maximum number of samples
        [~, maxCluster] = max(hist(clusterIdx, 1:maxClusters));

        % Create a binary mask based on the identified cluster
        clusterMask = clusterIdx == maxCluster;

        % Apply the binary mask to the original image
        clusteredFeatures = clusterMask;

        % Put the test features into variable 'test'
        test = testFeatures;

        % Classify the result based on the automatically selected cluster
        result = multisvm(Train_Feat(clusterIdx == maxCluster, :), Train_Label(clusterIdx == maxCluster), test);
    end

    function displayResult(class)
        % Visualize Results
        switch class
            case 0
                helpdlg('Downy Mildew', 'Result');
                disp('Downy Mildew');
            case 1
                helpdlg('Powdery Mildew', 'Result');
                disp('Powdery Mildew');
            case 2
                helpdlg('Black Rot', 'Result');
                disp('Black Rot');
            case 3
                helpdlg('Anthracnose', 'Result');
                disp('Anthracnose');
            otherwise
                helpdlg('Unknown Class', 'Result');
                disp('Unknown Class');
        end
    end

end

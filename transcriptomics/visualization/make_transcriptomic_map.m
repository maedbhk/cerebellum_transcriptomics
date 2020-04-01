% the code plots the MDTB-10-subRegions parcellation
% colors are given by the leaves of the dendrogram for 
% this same parcellation

addpath(genpath('/Users/maedbhking/Documents/cerebellum_transcriptomics'))
addpath(genpath('/Users/maedbhking/Documents/MATLAB'))

atlas_dir = '/Users/maedbhking/Documents/cerebellum_transcriptomics/data/external/atlas_templates';
cd(atlas_dir)

% map vol data to surface
atlas = "MDTB-10-subRegions";
C=suit_map2surf(fullfile(atlas_dir,sprintf('%s.nii', atlas)),'stats','mode');

% read in colors based on dendrogram 
T = readtable('MDTB-10-subRegions-transcriptomic-info.csv');

colors = table2array(T(:, [4,5,6])); 

for i=1:size(colors,1)
    cmapF(i,:) = colors(i,:);
end

suit_plotflatmap(C,'type','label','cmap',cmapF); % 'border',[]
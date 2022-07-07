% this code plots the MDTB-10-subRegions 
% parcellation (anterior portions are shaded)

atlas_dir = '/Users/maedbhking/Documents/cerebellum_transcriptomics/data/external/atlas_templates';
cd(atlas_dir)

% map vol data to surface
atlas = "MDTB-10-subRegions";
C=suit_map2surf(fullfile(atlas_dir,sprintf('%s.nii', atlas)),'stats','mode');

% read in color info
T = readtable('MDTB-10-subRegions-info.csv');

colors = table2array(T(:, [3,4,5])); 

for i=1:size(colors,1)
    cmapF(i,:) = colors(i,:)/255;
end

% get gifti coordinates for MDTB-10-subRegions map
G = gifti(fullfile(atlas_dir,sprintf('%s.label.gii', atlas)));

suit_plotflatmap(C,'type','label','cmap',cmapF); % 'border',[]
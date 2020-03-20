% I have forgotten how to code in matlab but this
% code is supposed to plot the sample coords from the
% cerebellum_transcriptomics project onto the cerebellar
% flatmap. Donor samples should have different colours. 
% code needs to be fixed.

cd('/Users/maedbhking/Downloads')

T = readtable('mni_coords_all_donors.csv');

colors = {'b','y','g','r','k','m'};

foci = table2array(T(:, [3,4,5,7]));

for i = 1:6,
    
    M=suit_map2surf(foci(foci(:,4)==i,1:3),'space','SPM');
    
    
    plot(M(:,1),M(:,2),'o','MarkerSize', colors{i});
    
end
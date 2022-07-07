% I have forgotten how to code in matlab but this
% code is supposed to plot the sample coords from the
% cerebellum_transcriptomics project onto the cerebellar
% flatmap. Donor samples should have different colours. 
% man, I hate matlab ...

data_dir = '/Users/maedbhking/Documents/cerebellum_transcriptomics/data/raw';

cd(data_dir)

T = readtable('mni_coords_all_donors.csv');

T.donor_id = string(T.donor_id); 

foci = table2array(T(:, [3,4,5,7]));
colors = table2array(T(:, [7,8,9,10])); 

% donors = ["donor10021", "donor9861", "donor14380", "donor15697", "donor15496", "donor12876"]; 

suit_plotflatmap([]);
hold on

for i = 1:6
    
    rgb=colors(colors(:,1)==i,2:4);
    M=suit_map2surf(foci(foci(:,4)==i,1:3),'space','SPM');
    
    donor = T.donor_id(T.donor_num==i);
   
    plot(M(:,1),M(:,2),'ko','MarkerSize',10,'MarkerFaceColor',[rgb(1,:)])
    
%     text(95,15*i,donor(1),'Color',[rgb(1,:)],'FontWeight','Bold','FontSize',10);
    
end
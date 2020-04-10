% make new atlas - Yeo-Buckner 7 and 17
% using spm_imcalc. there should be a way to 
% do this using nipype but I'm pressed for time
% so this will have to do 

atlas_dir = '/Users/maedbhking/Documents/cerebellum_transcriptomics/data/external/atlas_templates';
cd(atlas_dir)

num_regs = 7;

% create new Buckner-7 and 17 atlases
Vi = spm_vol(sprintf('Buckner-%d.nii', num_regs));
Vo = spm_read_vols(Vi);

Yy=zeros(1,Vi.dim(1)*Vi.dim(2)*Vi.dim(3));
for region=1:num_regs
    indx=find(Vo==region);
    Yy(1,indx)=round(region+num_regs);
end

Yy=reshape(Yy,[Vi.dim(1),Vi.dim(2),Vi.dim(3)]);
Yy(Yy==0)=NaN;

Vi.fname=sprintf('Buckner-%d-modified.nii', num_regs);
spm_write_vol(Vi,Yy);

% save out new concatenated Buckner and Yeo atlas
nam = {};
nam{1} = sprintf('Yeo-%d.nii', num_regs);
nam{2} = sprintf('Buckner-%d-modified.nii', num_regs);
spm_imcalc(nam, sprintf('Yeo-Buckner-%d.nii', num_regs), 'i1+i2', flags)


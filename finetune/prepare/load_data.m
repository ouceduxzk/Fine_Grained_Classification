load('cars_test_annos_withlabels.mat')
fileID = fopen('test_anno.txt','w');

for row = 1:8041
    fprintf(fileID,'%f %f %f %f %d %s\n', annotations(row).bbox_x1,annotations(row).bbox_y1, annotations(row).bbox_x2, annotations(row).bbox_y2, annotations(row).class, annotations(row).fname);
end
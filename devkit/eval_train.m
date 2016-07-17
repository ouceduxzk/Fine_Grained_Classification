function [accuracy, confusion_matrix] = eval_train(pred_fname)
% Evaluates training accuracy.
% Arguments:
%   pred_fname: Filename of prediction file for training. The required format
%     is described in the README.
% Returns:
%   accuracy: The accuracy on the training set.
%   confusion_matrix: The confusion matrix on the training set.

accuracy = [];
confusion_matrix = [];

train_data = load('cars_train_annos.mat');
train_annos = train_data.annotations;
train_classes = [train_annos.class];
unique_classes = unique(train_classes);

try
  preds = csvread(pred_fname);
catch err
  fprintf('Invalid file format for input file %s.', pred_fname);
  return
end


% Check whether predictions look sane
if numel(preds) ~= numel(train_classes)
  fprintf(['Given predictions have length %d but there are %d images ' ...
    'in the training set.\n'], numel(preds), numel(train_classes));
  return;
elseif any(~ismember(preds, unique_classes))
  bad_ind = find(~ismember(preds, unique_classes), 1);
  fprintf(['Predicted class for image %d is %d, which is an invalid ' ...
    'class.\n'], bad_ind, preds(bad_ind));
  return;
end

% Evaluate
accuracy = mean(preds(:) == train_classes(:));
confusion_matrix = confusionmat(train_classes(:), preds(:));

function cost = cautoCost2(p, data, num_conv, nt, patches, extra_hidden)
% cautoCost2.m
%   A cost function for training a 2D convolutional or translationally invariant autoencoder with an arbitrary number of logistic hidden layers
%   The accompanying gradient for optimization is dcautoCost2.m
%
%   Variable definitions: 
%      p -- network parameters; vector containing unrolled matrices of the following:
%           W: the convolution layer bias and weights (in that order); dimensions num_conv X size(data,2)+1
%           H: extra hidden layers bias and weights (bias and weights grouped by layer); dimensions extra_hidden(2) X (num_conv*length(patches)+1 OR extra_hidden(2)+1)
%           R: the construction layer bias and weights; dimensions size(data,2) X (num_conv*length(patches)+1 OR extra_hidden(2)+1)
%           Reshaping each into a vector and concatenating in the order of the list above is the definition of p. The column dimensionality is the dimension of the prior layer output plus a bias term.
%      data -- the data organized into dimensions samples X features
%      num_conv -- the number of neurons in the set of the convolution layer; each patch is convolved with the set (see patches below)
%      nt -- number of features along the x-axis of data (the data)
%      patches -- patches is a vector of structs that define the patching parameters; each element of patches is a patch (subset of adjacent elements) of the data; the struct parameters are:
%           x0: offset of patch along x-axis
%           y0: offset of patch along y-axis
%           Nx: patch length along x-axis
%           Ny: patch length along y-axis
%           Nx and Ny must be constant across all patches. Choosing an offset of 1 is equivalent to having no offset. A 0 offset runs but is invalid. Negative offsets allow for incomplete patches (or patches that begin outside the frame of the data).
%      extra_hidden -- a vector with two elements that define (1) the number of extra hidden layers and (2) the number of neurons in each extra hidden layer. An empty vector indicates no extra hidden layers.
%


    [nsamples, ndim] = size(data);
    nf = ndim/nt;
    num_hidden = patches(1).Nx*patches(2).Ny;

    W = p(1:num_conv*(num_hidden+1));
    W = reshape(W, [num_conv length(W)/num_conv]);    

    if isempty(extra_hidden) || extra_hidden(1) <= 0
        R = p(num_conv*(num_hidden+1)+1:num_conv*(num_hidden+1)+ndim*(num_conv*numel(patches)+1));
	R = reshape(R, [ndim length(R)/ndim]);
    else
        extra_layers = extra_hidden(1);
        extra_num = extra_hidden(2);
        H = cell(1,extra_layers);
        if extra_layers == 1
            index = num_conv*(patches(1).Nx*patches(2).Ny+1)+1:num_conv*(patches(1).Nx*patches(2).Ny+1)+extra_num*(num_conv*length(patches)+1);
            H{1} = p(index);
            H{1} = reshape(H{1}, [extra_num length(H{1})/extra_num]);
        else
            index = num_conv*(patches(1).Nx*patches(2).Ny+1)+1:num_conv*(patches(1).Nx*patches(2).Ny+1)+extra_num*(num_conv*length(patches)+1);                
            H{1} = p(index);
            H{1} = reshape(H{1}, [extra_num length(H{1})/extra_num]);
            for j=2:extra_layers
                index = index(end)+1:index(end)+extra_num*(extra_num+1);
                H{j} = p(index);
                H{j} = reshape(H{j}, [extra_num length(H{j})/extra_num]);
            end
        end
        index = index(end)+1:index(end)+ndim*(extra_num+1);
        R = p(index);
        R = reshape(R, [ndim length(R)/ndim]);
    end

    Z = zeros(size(W,1)*length(patches),nsamples);
    TZ = cell(1,length(patches));
    parfor k = 1:length(patches)
        Xind = zeros(patches(k).Ny, patches(k).Nx);
        ind = 1;
        yrange = patches(k).y0:patches(k).y0+patches(k).Ny-1;
        xrange = patches(k).x0:patches(k).x0+patches(k).Nx-1;
        for x=xrange
            Xind(:,ind) = (x-1)*nf+patches(k).y0:(x-1)*nf+patches(k).y0+patches(k).Ny-1;
            ind = ind+1;
        end
        Xind(yrange < 1 | yrange > nf, :) = -1;
        Xind(:, xrange < 1 | xrange > nt) = -1;
        Xind = Xind(:);
        tmp = W(:,2:end);
        tmp = tmp(:, Xind' > 0 & Xind' <= size(data,2));
        Xind(Xind <= 0 | Xind > size(data,2)) = [];
        TZ{k} = 1./(1+exp( repmat(W(:,1), [1 nsamples]) + (data(:,Xind)*tmp')' ));
    end
    for k = 1:length(patches)
        Z((k-1)*size(W,1)+1:k*size(W,1),:) = TZ{k};
    end
    
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        x = repmat(R(:,1), [1 nsamples]) + R(:,2:end)*Z;
    else
        Y = cell(1,extra_layers);
        tmp = H{1};
        Y{1} = 1./(1+exp( repmat(tmp(:,1), [1 nsamples]) + tmp(:,2:end)*Z ));
        for j=2:extra_layers
            tmp = H{j};
            Y{j} = 1./(1+exp( repmat(tmp(:,1), [1 nsamples]) + tmp(:,2:end)*Y{j-1} ));
        end
        x = repmat(R(:,1), [1 nsamples]) + R(:,2:end)*Y{extra_layers};
    end

    cost = 0.5*sum(sum((data-x').^2))/size(data,1);

end

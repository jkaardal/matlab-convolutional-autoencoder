function dcost = dcautoCost2(p, data, num_conv, nt, patches, extra_hidden)
% dcautoCost2.m
%    A gradient of the cost function for a convolutional or translationally invariant autoencoder
%    See the accompanying cost function cautoCost2.m for variable definitions.
%


    [nsamples, num_hidden] = size(data);
    nf = num_hidden/nt;

    W = p(1:num_conv*(patches(1).Nx*patches(2).Ny+1));
    W = reshape(W, [num_conv length(W)/num_conv]);
    
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        R = p(num_conv*(patches(1).Nx*patches(2).Ny+1)+1:num_conv*(patches(1).Nx*patches(2).Ny+1)+num_hidden*(num_conv*length(patches)+1));
        R = reshape(R, [num_hidden length(R)/num_hidden]);
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
        index = index(end)+1:index(end)+num_hidden*(extra_num+1);
        R = p(index);
        R = reshape(R, [num_hidden length(R)/num_hidden]);
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

    dcb = ((x'-data)/size(data,1))';
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        drb = R(:,2:end)'*dcb.*(Z.*(Z-1));
    else
        drb = R(:,2:end)'*dcb.*(Y{extra_layers}.*(Y{extra_layers}-1));
        dhb = cell(1,extra_layers);
        tmp = H{extra_layers};
        if extra_layers == 1
            dhb{1} = tmp(:,2:end)'*drb.*(Z.*(Z-1));
        else
            dhb{extra_layers} = tmp(:,2:end)'*drb.*(Y{extra_layers-1}.*(Y{extra_layers}-1));
            for j=extra_layers-1:-1:2
                tmp = H{j};
                dhb{j} = tmp(:,2:end)'*dhb{j+1}.*(Y{j-1}.*(Y{j-1}-1));
            end
            tmp = H{1};
            dhb{1} = tmp(:,2:end)'*dhb{2}.*(Z.*(Z-1));
        end
    end
        
    
    dR = zeros(size(R(:,2:end)));
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        parfor j=1:size(R,1)
            dR(j,:) = sum(Z.*repmat(dcb(j,:), [size(R,2)-1 1]),2)';
        end
    else
        parfor j=1:size(R,1)
            dR(j,:) = sum(Y{extra_layers}.*repmat(dcb(j,:), [size(R,2)-1 1]),2)';
        end
        dH = cell(1,extra_layers);
        if extra_layers == 1
            tmp = H{1};
            tmp = zeros(size(tmp(:,2:end)));
            parfor j=1:size(tmp,1)
                tmp(j,:) = sum(Z.*repmat(drb(j,:), [size(H{1},2)-1 1]),2)';
            end
            dH{1} = tmp;
        else
            tmp = H{extra_layers};
            tmp = zeros(size(tmp(:,2:end)));
            parfor j=1:size(tmp,1)
                tmp(j,:) = sum(Y{extra_layers-1}.*repmat(drb(j,:), [size(H{extra_layers},2)-1 1]),2)';
            end
            dH{extra_layers} = tmp;
            parfor k=extra_layers-1:-1:2
                tmp = H{k};
                tmp = zeros(size(tmp(:,2:end)));
                d = dhb{k+1};
                for j=1:size(tmp,1)
                    tmp(j,:) = sum(Y{k-1}.*repmat(d(j,:), [size(H{j},2)-1 1]),2)';
                end
                dH{k} = tmp;
            end
            tmp = H{1};
            tmp = zeros(size(tmp(:,2:end)));
            d = dhb{2};
            parfor j=1:size(tmp,1)
                tmp(j,:) = sum(Z.*repmat(d(j,:), [size(H{1},2)-1 1]),2)';
            end
            dH{1} = tmp;
        end
    end
    
    dW = zeros(size(W(:,2:end)));
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        for k = 1:length(patches)
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
            Sind = Xind;
            Sind(Sind < 1 | Sind > size(data,2)) = [];
            for j=1:size(W,1)
                dW(j,Xind > 0 & Xind <= size(data,2)) = dW(j,Xind > 0 & Xind <= size(data,2)) + sum(data(:,Sind)'.*repmat(drb((k-1)*size(W,1)+j,:), [length(Sind) 1]), 2)';
            end
        end
    else
        for k = 1:length(patches)
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
            Sind = Xind;
            Sind(Sind < 1 | Sind > size(data,2)) = [];
            d = dhb{1};
            for j=1:size(W,1)
                dW(j,Xind > 0 & Xind <= size(data,2)) = dW(j,Xind > 0 & Xind <= size(data,2)) + sum(data(:,Sind)'.*repmat(d((k-1)*size(W,1)+j,:), [length(Sind) 1]), 2)';
            end
        end
    end
    
    DR = [sum(dcb,2) dR];
    drb = sum(drb,2);
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        drb = reshape(drb, [length(drb)/length(patches) length(patches)]);
        drb = sum(drb,2);
        DW = [drb dW];
    else
        if extra_layers == 1
            DH = [drb dH{1}];
        else
            DH = [];
            for j=1:extra_layers-1
                tmp = [sum(dhb{j+1},2) dH{j}];
                DH = [DH reshape(tmp, [1 numel(tmp)])];
            end
            tmp = [drb dH{extra_layers}];
            DH = [DH reshape(tmp, [1 numel(tmp)])];
        end
        dhb{1} = sum(dhb{1},2);
        dhb{1} = reshape(dhb{1}, [length(dhb{1})/length(patches) length(patches)]);
        dhb{1} = sum(dhb{1},2);
        DW = [sum(dhb{1},2) dW];
    end
    
    DW = reshape(DW, [1 numel(DW)]);
    DR = reshape(DR, [1 numel(DR)]);
    
    if isempty(extra_hidden) || extra_hidden(1) <= 0
        dcost = [DW DR];
    else
        DH = reshape(DH, [1 numel(DH)]);
        dcost = [DW DH DR];
    end
    
end

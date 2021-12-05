% Run for data set

% clear
% clc

a1s = [0.005];

a2s = [8.5];

a3s = a2s;

figure('color','w')

for n1 = 1:length(a1s)
    a1 = a1s(n1);
    
    for n2 = 1:length(a2s)
        a2 = a2s(n2);
        a3 = a3s(n2);
         [u_ode,x,t] = AllenEQ(a1,a2,a3,1000);
%         u_data(:,:,n1, n2) = u;
        u = reshape(output',[1000,1000]);
%         num = length(a2s)*(n1-1) + n2;
        %         subplot(4, 3, num )
        pcolor(t,x,u')
        shading interp
        %         map = cbrewer('div','spectral',72);
        cmap = getPyPlot_cMap('Spectral_r');
        colormap(cmap)
        caxis([-1,1])
        xlim([0,1])
        ylim([-1,1])
        shading interp
        axis off
%         colorbar
%         h = colorbar;
%         ylabel(h, '\textrm{u}','Interpreter','latex')
        
%         xlabel('\textrm{t}','Interpreter','latex')
%         ylabel('\textrm{x}','Interpreter','latex')
        
%         set(gcf,'renderer','Painters')
        %         title(['a1 = ' num2str(a1s(n1)) , ',a2= ',num2str(a2s(n2))] )
    end
end

latex_fig(10, 1.5, 0.5)
print(gcf,'full_field_single_NN_0085_8.5_30_30_60.png','-dpng','-r300'); 

function latex_fig(font_size, f_width, f_height)
% font_size: the font size used in the paper;
% f_width: the figure width (in inches)
% f_height: the figure height (in inches)
font_rate=10/font_size;
set(gcf,'Position',[100   200   round(f_width*font_rate*144)   round(f_height*font_rate*144)])
end

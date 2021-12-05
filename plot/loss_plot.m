figure('color','w')
steps = 1:numel(train_loss);
% plot raw traning loss
a1 = plot(steps,log10(train_loss),'color',[128/255,177/255,211/255,0.1],'LineWidth',0.5);
% plot raw validation loss
hold on
a2 = plot(steps,log10(val_loss),'color',[251/255,128/255,114/255,0.01],'LineWidth',0.5);
% plot movmean training loss
a3 = plot(steps,movmean(log10(val_loss),1e3),'color',[228/255,26/255,28/255],'LineWidth',1.5);
a4 = plot(steps,movmean(log10(train_loss),1e3),'color',[55/255,126/255,184/255,0.5],'LineWidth',1.5);
legend([a3 a4],'\textrm{training loss}','\textrm{validation loss}','Interpreter','latex','fontsize',10)
xlabel('\textrm{steps}','Interpreter','latex','FontSize',10)
ylabel('\textrm{log(loss)}','Interpreter','latex','FontSize',10)
latex_fig(10, 3, 0.8)

function latex_fig(font_size, f_width, f_height)
% font_size: the font size used in the paper;
% f_width: the figure width (in inches)
% f_height: the figure height (in inches)
font_rate=10/font_size;
set(gcf,'Position',[100   200   round(f_width*font_rate*144)   round(f_height*font_rate*144)])
end

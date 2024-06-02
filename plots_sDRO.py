import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ALG.Utils import *
import numpy as np

DATA_LIMIT = 5000
PLOT_LIMIT = 150
for mu_y in [0.01]:
    data_name = 'gisette' + '_muy_' + str(mu_y) + f'_kappa_{1}_b_{200}'
    data_path = f'./result_data/{data_name}'
    for plot_part in ['x','y','z','loss','acc','lr_x','lr_y','primalF']:
        G = {}
        # G['GS-GDA-B,N=2'] = data_path +'/primal_line_search_N_2_AGDA'
        # G['GS-GDA-B,N=5'] = data_path +'/primal_line_search_N_5_AGDA'
        # G['GS-GDA-B,N=1'] = data_path +'/primal_line_search_N_1_AGDA'
        G['LS-GS-GDA'] = data_path +'/LS-GS-GDA'
        #G['LS-GS-GDA-S'] = data_path + '/LS-GS-GDA-S'
        #G['LS-GS-GDA-R'] = data_path +'/LS-GS-GDA-R'
        # G['LS-GS-GDA-S-R'] = data_path + '/LS-GS-GDA-S-R'
        # G['J-GDA'] = data_path +'/GDA'
        # G['GS-GDA'] = data_path +'/AGDA'
        G['TiAda'] = data_path +'/TiAda'


        plt.figure(dpi=150)
        fig, ax = plt.subplots()
        is_log = False
        C = 0.0  # value center for log s

        for alg_name, file_name in G.items():
            data_xLimit = DATA_LIMIT
            plot_xLimit = PLOT_LIMIT
            with open(file_name, "rb") as fp:  # Unpickling
                record = pickle.load(fp)
                # load x-axis data
                oracle_complexity_counter = min(record['oracle_complexity_counter'], key=len)
                sample_complexity_counter = min(record['sample_complexity_counter'], key=len)
                iter_counter = min(record['iter_counter'], key=len)
                epoch_counter = min(record['epoch_counter'], key=len)
                total_oracle_complexity_counter = min(record['total_oracle_complexity_counter'], key=len)
                total_sample_complexity_counter = min(record['total_sample_complexity_counter'], key=len)
                total_iter_counter = min(record['total_iter_counter'], key=len)
                total_epoch_counter = min(record['total_epoch_counter'], key=len)
                #counter = total_oracle_complexity_counter[:data_xLimit]
                counter = total_iter_counter[:data_xLimit]
                counter = total_epoch_counter[:DATA_LIMIT]
                data_xLimit = min(data_xLimit, len(counter))

                # load y-axis data
                valid_line_search = [i for i in range(len(record['acc']))]
                print(valid_line_search)

                acc = record['acc']
                acc = [acc[i][:data_xLimit] for i in valid_line_search]
                loss = [record['loss'][i][:data_xLimit] for i in valid_line_search]
                primal = [record['primalF'][i][:data_xLimit] for i in valid_line_search]
                error = [[1 - ele[i] for i in range(len(acc[0]))] for ele in acc]
                lr_x = record['lr_x']
                lr_y = record['lr_y']
                lr_x = [lr_x[i][:data_xLimit] for i in valid_line_search]
                lr_y = [lr_y[i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_sto_grad_x = [record['norm_square_sto_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_sto_grad_y = [record['norm_square_sto_grad_y'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_sto_grad_z = [[norm_sqaure_sto_grad_x[i][j] + norm_sqaure_sto_grad_y[i][j] for j in
                                           range(len(norm_sqaure_sto_grad_x[i]))] for i in
                                          range(len(norm_sqaure_sto_grad_x))]


                norm_sqaure_full_grad_x = [record['norm_square_full_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_full_grad_y = [record['norm_square_full_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_full_grad_z = [[norm_sqaure_full_grad_x[i][j] + norm_sqaure_full_grad_y[i][j] for j in
                                            range(len(norm_sqaure_full_grad_x[i]))] for i in
                                           range(len(norm_sqaure_full_grad_x))]



                # norm_sqaure_sto_grad_x = normlize_data(norm_sqaure_sto_grad_x)
                # norm_sqaure_sto_grad_y = normlize_data(norm_sqaure_sto_grad_y)
                # norm_sqaure_sto_grad_z = normlize_data(norm_sqaure_sto_grad_z)
                # norm_sqaure_full_grad_x = normlize_data(norm_sqaure_full_grad_x)
                # norm_sqaure_full_grad_y = normlize_data(norm_sqaure_full_grad_y)
                # norm_sqaure_full_grad_z = normlize_data(norm_sqaure_full_grad_z)
                contraction_times = record['contraction_times']
                b = record['config'][-1]['b']
                N = record['config'][-1]['N']

                if plot_part == 'x':
                    shadowplot(counter, norm_sqaure_full_grad_x, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=False, alg_name=alg_name)
                elif plot_part == 'y':
                    shadowplot(counter, norm_sqaure_full_grad_y, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=False, alg_name=alg_name)
                elif plot_part == 'z':
                    shadowplot(counter, norm_sqaure_full_grad_z, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=False, alg_name=alg_name,plot_part='z')
                elif plot_part == 'acc':
                    #error = [smooth_data(ele,100) for ele in error]
                    shadowplot(counter, error, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=False,
                               alg_name=alg_name)
                elif plot_part == 'loss':
                    shadowplot(counter, loss, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=False,
                               alg_name=alg_name)
                elif plot_part == 'primalF':
                    shadowplot(counter, primal, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name)
                elif plot_part == 'lr_x':
                    shadowplot(counter, lr_x, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=False,
                               alg_name=alg_name)
                elif plot_part == 'lr_y':
                    shadowplot(counter, lr_y, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=False,
                               alg_name=alg_name)

        if plot_part == 'x':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'y':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'z':
            if mu_y == 0.0001:
                plt.legend(fontsize=15, loc='lower left')
            else:
                plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'acc':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'loss' or plot_part == 'primalF':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'lr_x':
            plt.legend(fontsize=15, loc='lower right')
        elif plot_part == 'lr_y':
            plt.legend(fontsize=15, loc='lower right')

        # plt.legend(fontsize = 15, loc='lower left')
        # plt.legend(fontsize = 15,loc='upper left', bbox_to_anchor=(1.05, 1))
        #plt.xlabel("Number of Oracle Calls", fontsize=15)
        plt.xlabel("Number of epochs", fontsize=15)

        if plot_part == 'x':
            plt.ylabel(r"$\frac{||\nabla_x\mathcal{L}(x_k,y_k)||^2}{||\nabla_x\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'y':
            plt.ylabel(r"$\frac{||\nabla_y\mathcal{L}(x_k,y_k)||^2}{||\nabla_y\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'z':
            plt.ylabel(r"$\|\nabla\mathcal{L}(x_k,y_k)||^2$", fontsize=15)
            plt.ylabel(r"$\|G(x,y)||^2$", fontsize=15)
            #plt.ylabel(r"$\frac{||\nabla\mathcal{L}(x_k,y_k)||^2}{||\nabla\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'acc':
            plt.ylabel(r"Train Error", fontsize=15)
        elif plot_part == 'loss' or plot_part == 'primalF':
            plt.ylabel(r"Loss", fontsize=15)
        elif plot_part == 'lr_x':
            plt.ylabel(r"Stepsize $\tau$", fontsize=15)
        elif plot_part == 'lr_y':
            plt.ylabel(r"Stepsize $\sigma$", fontsize=15)

        # set label size here
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        # set x,y range here
        #
        # plt.ylim(1e-2,)
        plt.xlim(0,plot_xLimit)


        # set personalized axis scale here
        if plot_part == 'x':
            plt.yscale('log')
        elif plot_part == 'y':
            plt.yscale('log')
        elif plot_part == 'z':
            plt.yscale('log')
            #plt.ylim(0.3, 1.5)
        elif plot_part == 'acc':
            plt.ylim(0.04, 0.1)
            plt.xlim(1,160)
            plt.yscale('log')
            from matplotlib.ticker import ScalarFormatter

            ax.yaxis.set_major_formatter(ScalarFormatter())
        elif plot_part == 'loss' or plot_part == 'primalF':
            plt.ylim(0.15, 1)
            plt.yscale('log')
        elif plot_part == 'lr_x':
            plt.yscale('log')
        elif plot_part == 'lr_y':
            plt.yscale('log')

        # plt.xscale('log')
        if is_log:
            ax.set_yticklabels([round(np.exp(y) + C, 2) for y in ax.get_yticks()], fontsize=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.offsetText.set_visible(True)
        plt.grid()
        # set title here
        # plt.title('Qudradic_Bilinear_Obj',fontsize = 15)

        data_name_tmp = list(data_name)
        for i in range(len(data_name_tmp)):
            if data_name_tmp[i] == '.':
                data_name_tmp[i] = '_'

        plt.savefig(f'./figure/{"".join(data_name_tmp)}_{plot_part}.pdf', bbox_inches='tight', facecolor='w', dpi=150)

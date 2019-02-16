""" Running the CellGAN on Bodenmiller data. """

import argparse
import json
import time
import logging
from datetime import datetime as dt
import datetime
import tensorflow as tf
from sklearn.decomposition import PCA
import umap

from cellgan.lib.data_utils import load_fcs, get_fcs_filenames
from cellgan.lib.utils import get_filters, get_num_pooled, write_hparams_to_file
from cellgan.lib.utils import generate_subset, sample_z, compute_outlier_weights, build_logger
from cellgan.lib.utils import compute_frequency, assign_expert_to_subpopulation, compute_learnt_subpopulation_weights
from cellgan.model.model import CellGan
from cellgan.lib.plotting import *
from cellgan.experiments.bodenmiller.defaults import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('--in_dir', dest='input_dir', default=DATA_DIR,
                        help='Directory containing the input .fcs files')

    parser.add_argument('-o', '--out_dir', dest='output_dir', default=DEFAULT_OUT_DIR,
                        help='Directory where output will be generated.')

    parser.add_argument('--markers', nargs='+', type=str, default=DEFAULT_MARKERS,
                        help='List of markers')

    # data processing
    parser.add_argument('--inhibitor', default=DEFAULT_INHIBITOR, 
                        help='Inhibitor used for the experiment')

    parser.add_argument('--strength', default=DEFAULT_INHIB_STRENGTH, 
                        dest='inhib_strength', help='strength of inhibitor used.')

    parser.add_argument('--sub_limit', dest='subpopulation_limit', type=int, default=30,
                        help='Minimum number of cells to be called a subpopulation')

    parser.add_argument('--cofactor', dest='cofactor', type=int, default=5,
                        help='cofactor for the arcsinh transformation')

    # multi-cell input specific
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=64,
                        help='batch size used for training')

    parser.add_argument('-nc', '--ncell', dest='num_cells_per_input', type=int, default=100,
                        help='Number of cells per multi-cell input')

    # Generator Parameters
    parser.add_argument('-ns', '--noise_size', dest='noise_size', type=int, default=100,
                        help='Noise dimension for generator')

    parser.add_argument('--moe_sizes', dest='moe_sizes', nargs='+', type=int, default=[100, 100],
                        help='Sizes of the Mixture of Experts hidden layers')

    parser.add_argument('-e', '--experts', dest='num_experts', type=int,
                        help='Number of experts in the generator')

    parser.add_argument('-g', '--g_filters', dest='num_filters_generator', type=int, default=20,
                        help='Number of filters in conv layer of generator')

    parser.add_argument('--n_top', dest='num_top', default=1, type=int,
                        help='Number of experts used for generating each cell')

    parser.add_argument('--noisy_gating', default=True,
                        help='Whether to add noise to gating weights at train time')

    parser.add_argument('--noise_eps', default=1e-2, type=float, help='Noise threshold')

    parser.add_argument('--load_balancing', action='store_true',
                        help='Whether to add load balancing to Mixture of experts')

    parser.add_argument('--moe_loss_coef', default=1e-1, type=float,
                        help='Loss coefficient for mixture of experts loss')

    # Discriminator Parameters
    parser.add_argument('--num_cell_cnns', default=30, type=int,
                        help='Number of CellCnns in the ensemble')

    parser.add_argument('--d_filters_min', default=7, type=int,
                        help='Minimum number of filters for a CellCnn')

    parser.add_argument('--d_filters_max', default=10, type=int,
                        help='Maximum number of filters for a CellCnn')

    parser.add_argument('-cl1', '--coeff_l1', dest='coeff_l1', default=0,
                        type=float, help='Coefficient for l1 regularizer')

    parser.add_argument('-cl2', '--coeff_l2', dest='coeff_l2', default=1e-4,
                        type=float, help='Coefficient for l2 regularizer')

    parser.add_argument('--coeff_act', dest='coeff_act', default=0,
                        type=float, help='No clue what this is')

    parser.add_argument('--dropout', dest='dropout_prob', type=float, default=0.5,
                        help='Dropout probability')

    # Training Specific
    parser.add_argument('--subset', dest='subset_sample', default='outlier',
                        choices=['random', 'outlier'],
                        help='Whether to bias the subset selection towards rare subpopulations')

    parser.add_argument('--seed', default=None, type=int, help='Seed for reproducing results')

    parser.add_argument('-d_lr', '--disc_learning_rate', dest='disc_learning_rate', type=float,
                        default=2e-4, help='Learning rate for the discriminator')
    parser.add_argument('-g_lr', '--gen_learning_rate', dest='gen_learning_rate', type=float, 
                        default=2e-4, help='Learning rate for the generator')

    parser.add_argument('--num_critic', dest='num_critic', type=int, default=2,
                        help='Discriminator/Generator training Ratio')

    parser.add_argument('--beta_1', dest='beta_1', type=float, default=0.9,
                        help='beta_1 value for adam optimizer')

    parser.add_argument('--beta_2', dest='beta_2', type=float, default=0.999,
                        help='beta_2 value for adam optimizer')

    parser.add_argument('--type_gan', dest='type_gan', choices=['normal', 'wgan', 'wgan-gp'],
                        default='wgan', help='Type of GAN used for training')

    parser.add_argument('--clip_val', dest='clip_val', default=0.01,
                        help='clip discriminator values when using wgan')

    parser.add_argument('--init_method', dest='init_method',
                        choices=['xavier', 'zeros', 'normal'], default='xavier',
                        help='Initialization method for kernel and parameters')

    parser.add_argument('--reg_lambda', dest='reg_lambda', type=float,
                        default=10, help='reg_lambda value used in wgan-gp')

    parser.add_argument('--num_iter', dest='num_iterations', type=int,
                        default=10000, help='Number of iterations to run the GAN')

    # Testing, saving and plotting
    parser.add_argument('--num_samples', dest='num_samples', type=int,
                        help='Number of samples to generate while testing')

    parser.add_argument('--plot_every', type=int, default=500,
                        help='Add plots every n samples')

    parser.add_argument('--plot_heatmap', action='store_true', 
                        help='Whether to plot heatmap.')

    parser.add_argument('--save_model_every_n', action='store_true', 
                        help='Whether to save the model every time plots are added')

    parser.add_argument('--each_subpop', action='store_true',
                        help='Whether to plot expert vs each subpopulation')

    parser.add_argument('--real_vs_expert', action='store_true',
                        help='Whether to plot all real vs expert')

    args = parser.parse_args()

    each_subpop = False
    if args.each_subpop:
        each_subpop = True

    all_real_vs_expert = False
    if args.real_vs_expert:
        all_real_vs_expert = True

    # Setup the output directory
    experiment_name = dt.now().strftime("%d_%m_%Y-%H_%M_%S")
    output_dir = os.path.join(args.output_dir, args.inhibitor, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    markers_of_interest = args.markers
    inhibitor = args.inhibitor
    inhibitor_strength = args.inhib_strength
    fcs_savefile = os.path.join(output_dir, 'fcs.csv')
    markers_savefile = os.path.join(output_dir, 'markers.csv')

    fcs_files_of_interest = get_fcs_filenames(args.input_dir, inhibitor, inhibitor_strength)

    # Saving list of files and markers to output directory
    with open(fcs_savefile, 'w') as f:
        f.write(json.dumps(fcs_files_of_interest))
    with open(markers_savefile, 'w') as f:
        f.write(json.dumps(markers_of_interest))

    # Build logger
    cellgan_logger = build_logger(
        out_dir=output_dir, logging_format='%(message)s', level=logging.INFO)

    # Data Loading Steps
    cellgan_logger.info('Starting to load and process the .fcs files...')
    start_time = time.time()

    training_data, training_labels = load_fcs(
        fcs_files=fcs_files_of_interest,
        markers=markers_of_interest,
        args=args,
        logger=cellgan_logger
    )
    training_data = np.vstack(training_data)
    training_labels = np.concatenate(training_labels)

    cellgan_logger.info("Loading and processing completed.")
    cellgan_logger.info(
        'TIMING: File loading and processing took {} seconds \n'.format(
            datetime.timedelta(seconds=time.time() - start_time)))

    # Actual subpopulation weights
    weights_subpopulations = compute_frequency(labels=training_labels, weighted=True)

    # Sampling filters for CellCnn Ensemble
    cellgan_logger.info("Sampling filters for the CellCnn Ensemble...")
    d_filters = get_filters(
        num_cell_cnns=args.num_cell_cnns,
        low=args.d_filters_min,
        high=args.d_filters_max)
    cellgan_logger.info("Filters for the CellCnn Ensemble sampled.\n")

    # Sampling num pooled for CellCnn Ensemble
    cellgan_logger.info("Sampling number of cells to be pooled...")
    d_pooled = get_num_pooled(
        num_cell_cnns=args.num_cell_cnns,
        num_cells_per_input=args.num_cells_per_input)
    cellgan_logger.info("Number of cells to be pooled sampled.\n")

    num_subpopulations = len(np.unique(training_labels))
    num_experts = args.num_experts

    if not args.num_samples:
        num_samples = training_data.shape[0]
    else:
        num_samples = args.num_samples

    # Initialize CellGan
    cellgan_logger.info('Building CellGan...')

    model = CellGan(
        noise_size=args.noise_size,
        moe_sizes=args.moe_sizes,
        batch_size=args.batch_size,
        num_markers=len(markers_of_interest),
        num_experts=num_experts,
        g_filters=args.num_filters_generator,
        d_filters=d_filters,
        d_pooled=d_pooled,
        coeff_l1=args.coeff_l1,
        coeff_l2=args.coeff_l2,
        coeff_act=args.coeff_act,
        num_top=args.num_top,
        dropout_prob=args.dropout_prob,
        noisy_gating=args.noisy_gating,
        noise_eps=args.noise_eps,
        d_lr=args.disc_learning_rate,
        g_lr=args.gen_learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        reg_lambda=args.reg_lambda,
        clip_val=args.clip_val,
        train=True,
        init_method=args.init_method,
        type_gan=args.type_gan,
        load_balancing=args.load_balancing)

    cellgan_logger.info('CellGan built. \n')

    moe_in_size = model.generator.get_moe_input_size()

    model_hparams = {
        'noise_size': args.noise_size,
        'moe_sizes': [moe_in_size] + args.moe_sizes + [len(markers_of_interest)],
        'batch_size': args.batch_size,
        'num_markers': len(markers_of_interest),
        'num_experts': num_experts,
        'g_filters': args.num_filters_generator,
        'd_filters_min': args.d_filters_min,
        'd_filters_max': args.d_filters_max,
        'd_filters': d_filters.tolist(),
        'd_pooled': d_pooled.tolist(),
        'coeff_l1': args.coeff_l1,
        'coeff_l2': args.coeff_l2,
        'coeff_act': args.coeff_act,
        'num_top': args.num_top,
        'dropout_prob': args.dropout_prob,
        'noisy_gating': args.noisy_gating,
        'noise_eps': args.noise_eps,
        'd_lr': args.disc_learning_rate,
        'g_lr': args.gen_learning_rate,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'reg_lambda': args.reg_lambda,
        'clip_val': args.clip_val,
        'init_method': args.init_method,
        'type_gan': args.type_gan,
        'load_balancing': args.load_balancing,
        'num_critic': args.num_critic,
        'num_cell_per_input': args.num_cells_per_input,
        'num_cell_cnns': args.num_cell_cnns,
        'inhibitor_strength': inhibitor_strength,
        'subpopulation_limit': args.subpopulation_limit,
        'cofactor': args.cofactor

    }

    # Write hparams to text file (for reproducibility later)
    cellgan_logger.info('Saving hyperparameters...')
    write_hparams_to_file(out_dir=output_dir, hparams=model_hparams)
    cellgan_logger.info('Hyperparameters saved. \n')

    # Log data to output file
    cellgan_logger.info("Experiment Name: " + experiment_name)
    cellgan_logger.info("Inhibitor {} with strength {} ".format(inhibitor, inhibitor_strength))
    cellgan_logger.info("Starting our experiments with {} subpopulations".
                        format(num_subpopulations))
    cellgan_logger.info("Number of filters in the CellCnn Ensemble are: {}".format(d_filters))
    cellgan_logger.info(
        "number of cells pooled in the CellCnn Ensemble are: {} \n".format(
            d_pooled))

    # Training the Gan
    discriminator_loss = list()
    generator_loss = list()

    # Fit PCA object
    pca = PCA(n_components=2)
    pca = pca.fit(training_data)

    # Fit UMAP object
    um1 = umap.UMAP()
    um1 = um1.fit(training_data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for iteration in range(args.num_iterations):

            if args.subset_sample == 'outlier':
                subset_size = np.random.randint(low=20, high=50)
                outlier_scores = compute_outlier_weights(
                    inputs=training_data, method='q_sp', subset_size=subset_size)

            # Discriminator Training
            model.set_train(True)

            for _ in range(args.num_critic):
                if args.subset_sample == 'outlier':
                    real_batch, indices_batch = \
                        generate_subset(inputs=training_data, num_cells_per_input=args.num_cells_per_input,
                                        weights=outlier_scores, batch_size=args.batch_size,
                                        return_indices=True)

                elif args.subset_sample == 'random':
                    real_batch, indices_batch = \
                        generate_subset(inputs=training_data, num_cells_per_input=args.num_cells_per_input,
                                        weights=None, batch_size=args.batch_size, return_indices=True)

                else:
                    raise NotImplementedError('Subset sampling method {} is not '
                                              'supported'.format(args.subset_sample))

                noise_batch = sample_z(
                    batch_size=args.batch_size,
                    noise_size=args.noise_size,
                    num_cells_per_input=args.num_cells_per_input)

                if args.type_gan == 'wgan':
                    fetches = [model.d_solver, model.d_loss, model.clip_D]
                    feed_dict = {model.Z: noise_batch, model.X: real_batch}
                    _, d_loss, _ = sess.run(
                        fetches=fetches, feed_dict=feed_dict)

                elif args.type_gan == 'normal':
                    fetches = [model.d_solver, model.d_loss]
                    feed_dict = {model.Z: noise_batch, model.X: real_batch}
                    _, d_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

                else:
                    raise NotImplementedError(
                        'Support for wgan-gp not implemented yet.')

            # Generator training
            noise_batch = sample_z(
                batch_size=args.batch_size,
                noise_size=args.noise_size,
                num_cells_per_input=args.num_cells_per_input)

            fetches = [model.g_solver, model.g_loss, model.generator.moe_loss]
            feed_dict = {model.Z: noise_batch, model.X: real_batch}

            _, g_loss, moe_loss = sess.run(
                fetches=fetches, feed_dict=feed_dict)

            discriminator_loss.append(d_loss)
            generator_loss.append(g_loss)

            if iteration % args.plot_every == 0:
                model.set_train(False)

                frequency_sampled_batch = compute_frequency(
                    labels=training_labels[indices_batch], weighted=True)

                # Iteration number and losses
                cellgan_logger.info(
                    "We are at iteration: {}".format(iteration + 1))
                cellgan_logger.info("Discriminator Loss: {}".format(d_loss))
                cellgan_logger.info("Generator Loss: {}".format(g_loss))
                cellgan_logger.info(
                    "Load Balancing Loss: {} \n".format(moe_loss))

                # Actual weights & sampled weights
                cellgan_logger.info("The actual subpopulation weights are: {}".
                                    format(weights_subpopulations))
                cellgan_logger.info(
                    "Weights after outlier based sampling: {} \n".format(
                        frequency_sampled_batch))

                # Sample fake data for testing
                noise_sample = sample_z(
                    batch_size=1,
                    num_cells_per_input=num_samples,
                    noise_size=args.noise_size)

                fetches = [model.g_sample, model.generator.gates, model.generator.logits]
                feed_dict = {model.Z: noise_sample}

                fake_samples, gates, logits = sess.run(fetches=fetches, feed_dict=feed_dict)
                fake_samples = fake_samples.reshape(num_samples, len(markers_of_interest))
                fake_sample_experts = np.argmax(gates, axis=1)

                # Sample real data for testing
                real_samples, indices = generate_subset(
                    inputs=training_data,
                    num_cells_per_input=num_samples,
                    weights=None,
                    batch_size=1,
                    return_indices=True)
                real_samples = real_samples.reshape(num_samples, len(markers_of_interest))
                indices = np.reshape(indices, real_samples.shape[0])
                real_sample_subs = training_labels[indices]

                # Compute expert assignments based on KS test
                expert_assignments = \
                    assign_expert_to_subpopulation(real_data=real_samples, real_labels=real_sample_subs,
                                                   fake_data=fake_samples, expert_labels=fake_sample_experts,
                                                   num_experts=num_experts, num_subpopulations=num_subpopulations)

                # Compute learnt subpopulation weights
                learnt_subpopulation_weights = \
                    compute_learnt_subpopulation_weights(expert_labels=fake_sample_experts,
                                                         expert_assignments=expert_assignments,
                                                         num_subpopulations=num_subpopulations)

                cellgan_logger.info("The actual subpopulation weights are: {}".
                                    format(weights_subpopulations))
                cellgan_logger.info(
                    "The learnt subpopulation weights are: {} \n".format(learnt_subpopulation_weights))

                # Save loss plot
                cellgan_logger.info("Saving loss plot")
                plot_loss(
                    out_dir=output_dir,
                    disc_loss=discriminator_loss,
                    gen_loss=generator_loss)
                cellgan_logger.info("Loss plot saved. \n")

                # Plot marker distributions
                cellgan_logger.info("Adding marker distribution plots...")
                
                plot_marker_distributions(
                    out_dir=output_dir,
                    real_subset=real_samples,
                    fake_subset=fake_samples,
                    real_subset_labels=real_sample_subs,
                    fake_subset_labels=fake_sample_experts,
                    num_experts=num_experts,
                    num_markers=len(markers_of_interest),
                    num_subpopulations=num_subpopulations,
                    marker_names=markers_of_interest,
                    iteration=iteration,
                    logger=cellgan_logger,
                    zero_sub=True)
                
                cellgan_logger.info("Marker distribution plots added. \n")
                
                # PCA plot
                cellgan_logger.info("Adding PCA plots...")

                plot_pca(
                    out_dir=output_dir, pca_obj=pca, 
                    real_subset=real_samples,
                    fake_subset=fake_samples,
                    real_subset_labels=real_sample_subs,
                    fake_subset_labels=fake_sample_experts,
                    num_experts=num_experts,
                    num_subpopulations=num_subpopulations,
                    iteration=iteration,
                    logger=cellgan_logger,
                    zero_sub=True,
                    each_subpop=each_subpop, 
                    all_real_vs_expert=all_real_vs_expert)

                cellgan_logger.info("PCA plots added. \n")

                # UMAP plot

                cellgan_logger.info("Adding UMAP plots.")
                plot_umap(
                    out_dir=output_dir, umap_obj=um1,
                    real_subset=real_samples,
                    fake_subset=fake_samples,
                    real_subset_labels=real_sample_subs,
                    fake_subset_labels=fake_sample_experts,
                    num_experts=num_experts,
                    num_subpopulations=num_subpopulations,
                    iteration=iteration,
                    logger=cellgan_logger,
                    zero_sub=True,
                    each_subpop=each_subpop,
                    all_real_vs_expert=all_real_vs_expert)

                # Umap on fake data
                um2 = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
                um2.fit(fake_samples)
                transformed_real = um2.transform(real_samples)

                f = plt.figure(figsize=(20, 20))
                plt.scatter(transformed_real[:, 0], transformed_real[:, 1], c=real_sample_subs, label='Real-Data')
                plt.xlabel('UM1')
                plt.ylabel('UM2')
                plt.legend()
                f.tight_layout()
                plt.savefig(os.path.join(output_dir, str(iteration+1), "input_space_clustering.pdf"))
                plt.close()

                cellgan_logger.info("UMAP plots added. \n")

                # Plotting the heatmap of gating weights
                if args.plot_heatmap:
                    cellgan_logger.info("Adding Heatmap...")
                    plot_heatmap(out_dir=output_dir, iteration=iteration,
                                 logits=logits, fake_subset_labels=fake_sample_experts)
                    cellgan_logger.info("Heatmap added")

                # Save the model
                cellgan_logger.info("Saving the model...")
                saver = tf.train.Saver()
                model_path = os.path.join(output_dir, str(iteration+1), 'model.ckpt')
                save_path = saver.save(sess, model_path)
                cellgan_logger.info("Model saved.")

                cellgan_logger.info(
                    "########################################## \n")


if __name__ == '__main__':
    main()

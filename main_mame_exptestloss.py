
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.data_generator import ImageDataGenerator
# from masf_func_fc4_5_11 import MASF
import random
FLAGS = flags.FLAGS
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Dataset PACS
flags.DEFINE_string('method', 'MET', 'deepall/ET/MET')
# flags.DEFINE_bool('MET', True, 'mix or sole for ET')
flags.DEFINE_string('dataset', 'pacs_our', 'set the dataset of PACS/VLCS')
flags.DEFINE_string('loss', 'L_all', 'L_c/L_c+L_g/L_c+L_g+L_s')
flags.DEFINE_string('target_domain', 'NCH', 'set the target domain')
flags.DEFINE_string('dataroot', '', 'Root folder where PACS dataset is stored')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification.')

## Training option1
flags.DEFINE_integer('train_iterations', 5000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 120, 'number of images sampled per source domain')
flags.DEFINE_float('inner_lr', 5e-5, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 5e-5, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 1e-5, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 20, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')
flags.DEFINE_float('a0', 0.0, 'trade-off for meta_test task loss')
flags.DEFINE_float('a1', 1.0, 'trade-off for global loss')
flags.DEFINE_float('a2', 0.5, 'trade-off for metric loss')



## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './log/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('summary_interval', 30, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 20000, 'intervals to save model')
flags.DEFINE_integer('print_interval', 30, 'intervals to print out training info')
flags.DEFINE_integer('test_print_interval', 30, 'intervals to test the model')
flags.DEFINE_string('suffix', 'testlossvis_run1','suffix')

class_list = {'0': 'epi',
              '1': 'str'
              }

def train(model, saver, sess, exp_string, train_file_list, test_file, resume_itr=0):

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph,filename_suffix='_'+FLAGS.suffix)
    source_losses, target_losses, source_accuracies, target_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [],[],[]
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i], dataroot=FLAGS.dataroot, mode='training', \
                                         batch_size=int(1*FLAGS.meta_batch_size), num_classes=FLAGS.num_classes, shuffle=True)
            tr_data_list.append(tr_data)
            train_iterator_list.append(tf.data.Iterator.from_structure(tr_data.data.output_types,tr_data.data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())

        test_data = ImageDataGenerator(test_file, dataroot=FLAGS.dataroot, mode='inference', \
                                       batch_size=1, num_classes=FLAGS.num_classes, shuffle=False)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()

        ### testvis ###
        testvis_data = ImageDataGenerator(test_file[0], dataroot=FLAGS.dataroot, mode='training', \
                                       batch_size=int(1*FLAGS.meta_batch_size), num_classes=FLAGS.num_classes, shuffle=True)
        testvis_iterator = tf.data.Iterator.from_structure(testvis_data.data.output_types, testvis_data.data.output_shapes)
        testvis_next_batch = testvis_iterator.get_next()

    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        train_batches_per_epoch.append(int(np.floor(tr_data_list[i].data_size/int(1.5*FLAGS.meta_batch_size))))
    
    test_init_op = test_iterator.make_initializer(test_data.data)
    test_batches_per_epoch = int(np.floor(test_data.data_size / 1))

    testvis_init_op = testvis_iterator.make_initializer(testvis_data.data)
    testvis_batches_per_epoch = int(np.floor(testvis_data.data_size/int(1.5*FLAGS.meta_batch_size))*30)

    # Training begins
    best_test_acc = 0
    best_test_step = 0
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Sampling training and test tasks
        num_training_tasks = len(train_file_list)
        num_meta_train = num_training_tasks-1
        num_meta_test = num_training_tasks-num_meta_train  # as setting num_meta_test = 1

        # Randomly choosing meta train and meta test domains
        task_list = np.random.permutation(num_training_tasks)
        meta_train_index_list = task_list[:num_meta_train]
        meta_test_index_list = task_list[num_meta_train:]

        for i in range(len(train_file_list)):
            if (itr-resume_itr)%train_batches_per_epoch[i] == 0:
                sess.run(training_init_op[i])  # initialize training sample generator at itr=0

        if (itr-resume_itr)%testvis_batches_per_epoch == 0:
            sess.run(testvis_init_op)  

        # Sampling meta-train, meta-test data
        for i in range(num_meta_train):
            task_ind = meta_train_index_list[i]
            if i == 0:
                inputa, labela = sess.run(train_next_list[task_ind])
            elif i == 1:
                inputa1, labela1 = sess.run(train_next_list[task_ind])
            else:
                raise RuntimeError('check number of meta-train domains.')

        for i in range(num_meta_test):
            task_ind = meta_test_index_list[i]
            if i == 0:
                inputb, labelb = sess.run(train_next_list[task_ind])
            else:
                raise RuntimeError('check number of meta-test domains.')

        # to avoid a certain un-sampled class affect stability of of global class alignment
        # i.e., mask-out the un-sampled class from computing kd-loss
        sampledb = np.unique(np.argmax(labelb, axis=1))
        sampleda = np.unique(np.argmax(labela, axis=1))
        bool_indicator_b_a = [0.0] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
            # only count class that are sampled in both source domains
            if (i in sampledb) and (i in sampleda):
                bool_indicator_b_a[i] = 1.0

        sampledb = np.unique(np.argmax(labelb, axis=1))
        sampleda1 = np.unique(np.argmax(labela1, axis=1))
        bool_indicator_b_a1 = [0.0] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
            if (i in sampledb) and (i in sampleda1):
                bool_indicator_b_a1[i] = 1.0
        
        inner_lr = FLAGS.inner_lr
        outer_lr = FLAGS.outer_lr

        if FLAGS.method=='deepall':
            # from masf_func_deepall import MASF
            labela=labela[:FLAGS.meta_batch_size]
            labela1=labela1[:FLAGS.meta_batch_size]
            inputa=inputa[:FLAGS.meta_batch_size]
            inputa1=inputa1[:FLAGS.meta_batch_size]


            feed_dict = {model.inputa: inputa, model.labela: labela, \
                        model.inputa1: inputa1, model.labela1: labela1, \
                        model.inputb: inputb, model.labelb: labelb, \
                        model.KEEP_PROB: 0.5,model.inner_lr:inner_lr,model.outer_lr:outer_lr}

            output_tensors = [ model.task_train_op]
            output_tensors.extend([model.summ_op, model.source_loss, model.source_accuracy]) #model.summ_op在main中定义的
            _,  summ_writer, source_loss, source_accuracy = sess.run(output_tensors, feed_dict)

            source_losses.append(source_loss)
            source_accuracies.append(source_accuracy)

            if itr % FLAGS.print_interval == 0:
                print('---'*10 +f'lr=({inner_lr,outer_lr})'+ '\n%s' % (exp_string+'_'+FLAGS.suffix))
                print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
                print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
                source_losses, target_losses = [], []

        else: 
            # from masf_func_mame import MASF
            part = FLAGS.meta_batch_size // 3
            if FLAGS.method=='MET':
                sum = 10
                part_c = random.sample([0,3,5,7,10],1)[0]
                part_b = random.randint(0, sum - part_c)
                part_a = sum - part_c - part_b

                part_a = int(FLAGS.meta_batch_size * part_a / sum)
                part_b = int(FLAGS.meta_batch_size * part_b / sum)
                part_c = int(FLAGS.meta_batch_size - part_a - part_b)
   
                input_target_1 = np.concatenate((inputa[:part_a], inputa1[:part_b], inputb[:part_c]), axis=0)
                label_target_1 = np.concatenate((labela[:part_a], labela1[:part_b], labelb[:part_c]), axis=0)
            elif FLAGS.method=='ET':
                input_target_1 = inputb
                label_target_1 = labelb

            input_group = np.concatenate((inputa[:part], inputa1[:part], inputb[:part]), axis=0)
            label_group = np.concatenate((labela[:part], labela1[:part], labelb[:part]), axis=0)
            group_list = np.sum(label_group, axis=0)
            label_groupm = np.argmax(label_group, axis=1)  # transform one-hot labels into class-wise integer
            label_am = np.argmax(labela, axis=1)  # transform one-hot labels into class-wise integer
            label_a1m = np.argmax(labela1, axis=1)  # transform one-hot labels into class-wise integer
            label_bm = np.argmax(label_target_1, axis=1)  # transform one-hot labels into class-wise integer

            labela=labela[:FLAGS.meta_batch_size]
            labela1=labela1[:FLAGS.meta_batch_size]
            inputa=inputa[:FLAGS.meta_batch_size]
            inputa1=inputa1[:FLAGS.meta_batch_size]

            feed_dict = {
                        #model.domaina: task_list[0], model.domaina1: task_list[1], model.domainb: task_list[-1],\
                        model.inputa: inputa, model.labela: labela, \
                        model.inputa1: inputa1, model.labela1: labela1, \
                        model.inputb: input_target_1, model.labelb: label_target_1, \
                        # model.label_am: label_am, model.label_a1m: label_a1m, model.label_bm:label_bm, \
                        # model.input_group: input_group, model.label_group: label_group, model.label_groupm:label_groupm,\
                        model.bool_indicator_b_a: bool_indicator_b_a, model.bool_indicator_b_a1: bool_indicator_b_a1,
                        model.KEEP_PROB: 0.5, model.inner_lr:inner_lr,model.outer_lr:outer_lr}

            output_tensors = [ model.task_train_op, model.meta_train_op]
            output_tensors.extend([model.summ_op, model.global_loss, model.source_loss, model.source_accuracy, model.metric_loss, model.sm_loss]) #model.summ_op在main中定义的
            _, _,  summ_writer, global_loss, source_loss, source_accuracy, metric_loss, sm_loss = sess.run(output_tensors, feed_dict)

            source_losses.append(source_loss)
            source_accuracies.append(source_accuracy)
        
            if itr % FLAGS.print_interval == 0:
                print('---'*10 +f'lr=({inner_lr,outer_lr})'+ '\n%s' % (exp_string+'_'+FLAGS.suffix))
                print('number of samples per category:', group_list)
                print('global loss: %.7f' % global_loss)
                print('metric_loss: %.7f ' % metric_loss)
                print('sm_loss: %.7f ' % sm_loss)
                print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
                print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
                source_losses, target_losses = [], []

        if itr % FLAGS.summary_interval == 0 and FLAGS.log:
            train_writer.add_summary(summ_writer, itr)

        if (itr!=0) and itr % FLAGS.save_interval == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        if itr%30==0:
            inputb, labelb = sess.run(testvis_next_batch)
            if FLAGS.method=='deepall':
                # from masf_func_deepall import MASF
                feed_dict = {model.inputa: inputa, model.labela: labela, \
                            model.inputa1: inputa1, model.labela1: labela1, \
                            model.inputb: inputb, model.labelb: labelb, \
                            model.KEEP_PROB: 0.5,model.inner_lr:inner_lr,model.outer_lr:outer_lr}

                # output_tensors = [ model.task_train_op]
                # output_tensors.extend([model.summ_op, model.source_loss, model.source_accuracy]) #model.summ_op在main中定义的
                # _,  summ_writer, source_loss, source_accuracy = sess.run(output_tensors, feed_dict)

                # source_losses.append(source_loss)
                # source_accuracies.append(source_accuracy)

                # if itr % FLAGS.print_interval == 0:
                #     print('---'*10 +f'lr=({inner_lr,outer_lr})'+ '\n%s' % (exp_string+'_'+FLAGS.suffix))
                #     print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
                #     print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
                #     source_losses, target_losses = [], []

            else: 
                # from masf_func_mame import MASF
                part = FLAGS.meta_batch_size // 3
                # if FLAGS.method=='MET':
                #     sum = 10
                #     part_c = random.sample([0,3,5,7,10],1)[0]
                #     part_b = random.randint(0, sum - part_c)
                #     part_a = sum - part_c - part_b

                #     part_a = int(FLAGS.meta_batch_size * part_a / sum)
                #     part_b = int(FLAGS.meta_batch_size * part_b / sum)
                #     part_c = int(FLAGS.meta_batch_size - part_a - part_b)
    
                #     input_target_1 = np.concatenate((inputa[:part_a], inputa1[:part_b], inputb[:part_c]), axis=0)
                #     label_target_1 = np.concatenate((labela[:part_a], labela1[:part_b], labelb[:part_c]), axis=0)
                # elif FLAGS.method=='ET':
                #     input_target_1 = inputb
                #     label_target_1 = labelb

                input_target_1 = inputb
                label_target_1 = labelb

                # input_group = np.concatenate((inputa[:part], inputa1[:part], inputb[:part]), axis=0)
                # label_group = np.concatenate((labela[:part], labela1[:part], labelb[:part]), axis=0)
                # group_list = np.sum(label_group, axis=0)
                # label_groupm = np.argmax(label_group, axis=1)  # transform one-hot labels into class-wise integer
                # label_am = np.argmax(labela, axis=1)  # transform one-hot labels into class-wise integer
                # label_a1m = np.argmax(labela1, axis=1)  # transform one-hot labels into class-wise integer
                # label_bm = np.argmax(label_target_1, axis=1)  # transform one-hot labels into class-wise integer

                # labela=labela[:FLAGS.meta_batch_size]
                # labela1=labela1[:FLAGS.meta_batch_size]
                # inputa=inputa[:FLAGS.meta_batch_size]
                # inputa1=inputa1[:FLAGS.meta_batch_size]

                feed_dict = {
                            #model.domaina: task_list[0], model.domaina1: task_list[1], model.domainb: task_list[-1],\
                            model.inputa: inputa, model.labela: labela, \
                            model.inputa1: inputa1, model.labela1: labela1, \
                            model.inputb: input_target_1, model.labelb: label_target_1, \
                            # model.label_am: label_am, model.label_a1m: label_a1m, model.label_bm:label_bm, \
                            # model.input_group: input_group, model.label_group: label_group, model.label_groupm:label_groupm,\
                            model.bool_indicator_b_a: bool_indicator_b_a, model.bool_indicator_b_a1: bool_indicator_b_a1,
                            model.KEEP_PROB: 0.5, model.inner_lr:inner_lr,model.outer_lr:outer_lr}

                summ_writer = sess.run(model.summ_op,feed_dict)

                # output_tensors = [ model.task_train_op, model.meta_train_op]
                # output_tensors.extend([model.summ_op, model.global_loss, model.source_loss, model.source_accuracy, model.metric_loss, model.sm_loss]) #model.summ_op在main中定义的
                # _, _,  summ_writer, global_loss, source_loss, source_accuracy, metric_loss, sm_loss = sess.run(output_tensors, feed_dict)

                output_tensors = [model.ori_global_loss, model.source_loss, model.source_accuracy, model.ori_metric_loss, model.ori_sm_loss] #model.summ_op在main中定义的
                ori_global_loss, source_loss, source_accuracy, ori_metric_loss, ori_sm_loss = sess.run(output_tensors, feed_dict)

                if FLAGS.suffix:
                    testlossvis_file = os.path.join(FLAGS.logdir,exp_string,f'testlossvis_{FLAGS.suffix}.txt')
                else:
                    testlossvis_file = os.path.join(FLAGS.logdir,exp_string,'testlossvis.txt')

                with open(testlossvis_file, 'a') as fle:
                    # fle.write('Unseen Target Validation results: Iteration %d, Loss: %f, Accuracy: %f \n' %(itr, test_loss/test_count, test_acc))
                    fle.write(f'Unseen Target Validation results: Iteration {itr:d}, ori_global: {ori_global_loss:.5f}, ori_metric:  {ori_metric_loss:.5f}, ori_sm: {ori_sm_loss:.5f}, source: {source_loss:.5f}, source_acc: {source_accuracy:.5f} \n')


                # source_losses.append(source_loss)
                # source_accuracies.append(source_accuracy)

                # if itr % FLAGS.print_interval == 0:
                #     print('---'*10 +f'lr=({inner_lr,outer_lr})'+ '\n%s' % (exp_string+'_'+FLAGS.suffix))
                #     print('number of samples per category:', group_list)
                #     print('global loss: %.7f' % global_loss)
                #     print('metric_loss: %.7f ' % metric_loss)
                #     print('sm_loss: %.7f ' % sm_loss)
                #     print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
                #     print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
                #     source_losses, target_losses = [], []


        # Testing periodically
        class_accs = [0.0] * FLAGS.num_classes
        class_samples = [0.0] * FLAGS.num_classes
        if itr % FLAGS.test_print_interval == 0:
            test_acc, test_loss, test_count = 0.0, 0.0, 0.0
            sess.run(test_init_op) # initialize testing data generator
            for it in range(test_batches_per_epoch):
                test_input, test_label = sess.run(test_next_batch)
                feed_dict = {model.test_input: test_input, model.test_label: test_label, model.KEEP_PROB: 1.}
                output_tensors = [model.test_loss, model.test_acc]
                result = sess.run(output_tensors, feed_dict)
                test_loss += result[0]
                test_acc += result[1]
                test_count += 1
                this_class = np.argmax(test_label, axis=1)[0]
                class_accs[this_class] += result[1] # added for debug
                class_samples[this_class] += 1
            test_acc = test_acc/test_count
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_step = itr
                if FLAGS.suffix:
                    saver.save(sess, FLAGS.logdir + '/' + exp_string + f'/model_{FLAGS.suffix}_itr' + str(itr) + f'_acc{best_test_acc:.5f}',latest_filename=f'checkpoint_{FLAGS.suffix}')
                else:
                    saver.save(sess, FLAGS.logdir + '/' + exp_string + f'/model_itr' + str(itr) + f'_acc{best_test_acc:.5f}')
  

            print('Unseen Target Validation results: Iteration %d, Loss: %f, Accuracy: %f' %(itr, test_loss/test_count, test_acc))
            print('Current best dice {:.5f}, best step {} '.format(best_test_acc,best_test_step))   

            if FLAGS.suffix:
                eval_file = os.path.join(FLAGS.logdir,exp_string,f'eval_{FLAGS.suffix}.txt')
            else:
                eval_file = os.path.join(FLAGS.logdir,exp_string,'eval.txt')

            with open(eval_file, 'a') as fle:
                fle.write('Unseen Target Validation results: Iteration %d, Loss: %f, Accuracy: %f \n' %(itr, test_loss/test_count, test_acc))

def main():

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    filelist_root = './JMI/'  # path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line
    source_list = ['VGH', 'NKI', 'IHC', 'NCH']
    source_list.remove(FLAGS.target_domain)

    exp_string = 'mame_'  + FLAGS.target_domain + os.sep #+ '_' + FLAGS.loss + '_' + FLAGS.target_domain + os.sep 
    if FLAGS.method=='deepall':
        from masf_func_deepall import MASF
        exp_string = exp_string+'deepall_'
    else:
        from masf_func_mame import MASF
        exp_string = exp_string+ FLAGS.method+'['+str(FLAGS.a0)+','+str(FLAGS.a1)+','+str(FLAGS.a2)+']_'

    exp_string = exp_string+'mbs_' + str(FLAGS.meta_batch_size) + \
                 '.inner' + str(FLAGS.inner_lr) + '.outer' + str(FLAGS.outer_lr) + '.clipNorm' + str(
        FLAGS.gradients_clip_value) + \
                 '.metric' + str(FLAGS.metric_lr) + '.margin' + str(FLAGS.margin)
    
    # if FLAGS.suffix:
    #     exp_string = exp_string + '_'+FLAGS.suffix

    # Constructing model
    model = MASF()
    model.construct_model_train()
    model.construct_model_test()
    
    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),max_to_keep=3)
    sess = tf.InteractiveSession()
    output_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(output_var)


    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    
    print('Loading pretrained weights')
    model.load_initial_weights(sess)
    tf.get_default_graph()

    resume_itr = 0
    checkpoint_dir = ''
    #checkpoint_dir = './log/mame_NCH/deepall_mbs_120.inner5e-05.outer5e-05.clipNorm2.0.metric1e-05.margin20.0/'
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(checkpoint_dir)
        if model_file:
            # ind1 = model_file.index('model')
            # resume_itr = int(model_file[ind1+5:])
            resume_itr = int(model_file.split('itr')[-1].split('_')[0])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    train_file_list = [os.path.join(filelist_root, source_domain+'_list.txt') for source_domain in source_list]
    test_file_list = [os.path.join(filelist_root, FLAGS.target_domain+'_list.txt')]
    train(model, saver, sess, exp_string, train_file_list, test_file_list, resume_itr)

if __name__ == "__main__":
    main()

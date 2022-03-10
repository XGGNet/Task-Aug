
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.data_generator_seg import ImageDataGenerator
# from masf_func_fc4_5_11 import MASF
import random
from utils.utils import masked_np_dice, np_dice,post
import cv2
import SimpleITK as sitk
from glob import glob
from tensorflow.python import pywrap_tensorflow
import SimpleITK

FLAGS = flags.FLAGS
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Dataset PACS
flags.DEFINE_string('method', 'deepall', 'deepall/ET/MET')
flags.DEFINE_string('dataset', 'pacs_our', 'set the dataset of PACS/VLCS')
flags.DEFINE_string('loss', 'L_all', 'L_c/L_c+L_g/L_c+L_g+L_s')
flags.DEFINE_string('target_domain', 'BTCV', 'set the target domain from [BTCV,CHAOS,LITS,IRCAD]')
flags.DEFINE_string('dataroot', '', 'Root folder where PACS dataset is stored')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification.')

## Training option1
flags.DEFINE_integer('train_iterations', 10000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 120//2, 'number of images sampled per source domain')
flags.DEFINE_float('inner_lr', 5e-5, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 5e-5, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 1e-5, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 20, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')
flags.DEFINE_float('a0', 1.0, 'trade-off for meta_test task loss 0.0')
flags.DEFINE_float('a1', 0.0, 'trade-off for global loss 1.0')
flags.DEFINE_float('a2', 0.0, 'trade-off for metric loss 0.5')


## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './log/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('eval', False, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('summary_interval', 30, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 20000, 'intervals to save model')
flags.DEFINE_integer('print_interval', 30, 'intervals to print out training info')
flags.DEFINE_integer('test_print_interval', 30, 'intervals to test the model')
flags.DEFINE_integer('resize',128, '512 to 256 or 128')
# flags.DEFINE_string('suffix', 'flip(ud+lr)_128_lrld','suffix')
flags.DEFINE_string('suffix', 'warmup_mask2','suffix')

class_list = {'0': 'others',
              '1': 'liver'
              }

def train(model, saver, sess, exp_string, train_file_list, test_file, resume_itr=0):

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph,filename_suffix='_'+FLAGS.suffix)
    # modelsave_dir = FLAGS.logdir + '/' + exp_string 
    # if FLAGS.suffix:
    #     modelsave_dir = modelsave_dir + f'/{FLAGS.suffix}'
    #     if not os.path.exists(modelsave_dir):
    #         os.makedirs(modelsave_dir)
    source_losses, target_losses, source_accuracies, target_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [],[],[]
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i], dataroot=FLAGS.dataroot, mode='training', \
                                         batch_size=int(1*FLAGS.meta_batch_size), num_classes=FLAGS.num_classes, resize=FLAGS.resize, shuffle=True)
            tr_data_list.append(tr_data)
            train_iterator_list.append(tf.data.Iterator.from_structure(tr_data.data.output_types,tr_data.data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())
        if isinstance(test_file,list):
            test_file = test_file[0]
        test_data = ImageDataGenerator(test_file, dataroot=FLAGS.dataroot, mode='inference', \
                                       batch_size=1, num_classes=FLAGS.num_classes, resize=FLAGS.resize, shuffle=False)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()

    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        train_batches_per_epoch.append(int(np.floor(tr_data_list[i].data_size/int(1.5*FLAGS.meta_batch_size))))

    test_init_op = test_iterator.make_initializer(test_data.data)
    test_batches_per_epoch = int(np.floor(test_data.data_size / 1))

    # Training begins
    best_test_acc = 0
    best_test_dice = 0
    best_test_masked_dice = 0
    best_test_step = 0
    for itr in range(resume_itr, FLAGS.train_iterations):

        if not FLAGS.eval:
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

            # Sampling meta-train, meta-test data
            for i in range(num_meta_train):
                task_ind = meta_train_index_list[i]
                if i == 0:
                    inputa, labela, maska = sess.run(train_next_list[task_ind])
                elif i == 1:
                    inputa1, labela1, maska1 = sess.run(train_next_list[task_ind])
                else:
                    raise RuntimeError('check number of meta-train domains.')

            for i in range(num_meta_test):
                task_ind = meta_test_index_list[i]
                if i == 0:
                    inputb, labelb, maskb = sess.run(train_next_list[task_ind])
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
            
            # inner_lr = FLAGS.inner_lr
            # outer_lr = FLAGS.outer_lr*(1-(itr//1000)/10)

            inner_lr = FLAGS.inner_lr
            outer_lr = FLAGS.outer_lr
            
            if FLAGS.method=='deepall':
                # from masf_func_deepall import MASF
                labela=labela[:FLAGS.meta_batch_size]
                labela1=labela1[:FLAGS.meta_batch_size]
                inputa=inputa[:FLAGS.meta_batch_size]
                inputa1=inputa1[:FLAGS.meta_batch_size]

                feed_dict = {model.inputa: inputa, model.labela: labela, model.maska: maska,\
                            model.inputa1: inputa1, model.labela1: labela1, model.maska1: maska1,\
                            model.inputb: inputb, model.labelb: labelb, model.maskb: maskb,\
                            model.KEEP_PROB: 0.5, model.inner_lr:inner_lr,model.outer_lr:outer_lr}

                output_tensors = [ model.task_train_op]
                output_tensors.extend([model.summ_op, model.source_loss, model.source_accuracy])
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
                    mask_target_1 = np.concatenate((maska[:part_a], maska1[:part_b], maskb[:part_c]), axis=0)
                elif FLAGS.method=='ET':
                    input_target_1 = inputb
                    label_target_1 = labelb
                    mask_target_1 = maskb

                input_group = np.concatenate((inputa[:part], inputa1[:part], inputb[:part]), axis=0)
                label_group = np.concatenate((labela[:part], labela1[:part], labelb[:part]), axis=0)
                group_list = np.sum(label_group, axis=(0,1,2))
                label_groupm = np.argmax(label_group, axis=-1)  # transform one-hot labels into class-wise integer
                label_am = np.argmax(labela, axis=-1)  # transform one-hot labels into class-wise integer
                label_a1m = np.argmax(labela1, axis=-1)  # transform one-hot labels into class-wise integer
                label_bm = np.argmax(label_target_1, axis=-1)  # transform one-hot labels into class-wise integer

                labela=labela[:FLAGS.meta_batch_size]
                labela1=labela1[:FLAGS.meta_batch_size]
                inputa=inputa[:FLAGS.meta_batch_size]
                inputa1=inputa1[:FLAGS.meta_batch_size]

                # feed_dict = {model.domaina: task_list[0], model.domaina1: task_list[1], model.domainb: task_list[-1],\
                #             model.inputa: inputa, model.labela: labela, model.maska: maska, \
                #             model.inputa1: inputa1, model.labela1: labela1, model.maska1: maska1, \
                #             model.inputb: input_target_1, model.labelb: label_target_1, model.maskb: mask_target_1, \
                #             model.label_am: label_am, model.label_a1m: label_a1m, model.label_bm:label_bm, \
                #             model.input_group: input_group, model.label_group: label_group, model.label_groupm:label_groupm,\
                #             model.bool_indicator_b_a: bool_indicator_b_a, model.bool_indicator_b_a1: bool_indicator_b_a1,
                #             model.KEEP_PROB: 0.5,model.inner_lr:inner_lr,model.outer_lr:outer_lr}

                feed_dict = {
                            model.inputa: inputa, model.labela: labela, model.maska: maska, \
                            model.inputa1: inputa1, model.labela1: labela1, model.maska1: maska1, \
                            model.inputb: input_target_1, model.labelb: label_target_1, model.maskb: mask_target_1, \
                            model.bool_indicator_b_a: bool_indicator_b_a, model.bool_indicator_b_a1: bool_indicator_b_a1,
                            model.KEEP_PROB: 0.5,model.inner_lr:inner_lr,model.outer_lr:outer_lr}

                output_tensors = [ model.task_train_op, model.meta_train_op]
                output_tensors.extend([model.summ_op, model.global_loss, model.source_loss, model.source_accuracy, model.metric_loss, model.sm_loss])
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

        # Testing periodically
        class_accs = [0.0] * FLAGS.num_classes
        class_samples = [0.0] * FLAGS.num_classes
        if itr % FLAGS.test_print_interval == 0:
            test_acc, test_loss, test_count, test_dice,test_masked_dice = 0.0, 0.0, 0.0, 0.0, 0.0
            y_true, y_pred, x_input = [],[],[]
            # y_true2, y_pred2 = [],[]
            sess.run(test_init_op) # initialize testing data generator
            current_case = None
            case_index = []
            cs_cnt = 0
            for it in range(int(test_batches_per_epoch)):
                test_input, test_label,test_mask,test_filename = sess.run(test_next_batch)
                test_case = test_filename[0].decode().split('/')[-2]
                test_slice = int(test_filename[0].decode().split('/')[-1].split('.')[0].replace('image',''))
                if current_case==None:
                    current_case = test_case
                    cs_cnt = 1
                    case_index.append(0)
                else:
                    if test_case==current_case:
                        cs_cnt = cs_cnt+1
                    else:
                        case_index.append(cs_cnt)
                        current_case = test_case
                        cs_cnt = cs_cnt+1
                        
                feed_dict = {model.test_input: test_input, model.test_label: test_label,model.test_mask: test_mask, model.KEEP_PROB: 1.}
                output_tensors = [model.test_loss, model.test_acc, model.dice_slice, model.y_true, model.y_pred, model.y_prob]
                result = sess.run(output_tensors, feed_dict)
                test_loss += result[0]
                test_acc += result[1]
                # test_dice += result[2]

                # y_true.append(result[3])
                # y_pred.append(result[4])
                # x_input.append(test_input[...,0])
                y_true.append(result[3])
                y_pred.append(result[4])
                x_input.append(test_input[...,0])

                # label = cv2.resize(result[3][0],(512,512),interpolation=cv2.INTER_NEAREST)
                # y_true2.append(label)
                # pred = cv2.resize(result[5][0],(512,512))      
                # y_pred2.append(pred)

                # if it<=test_batches_per_epoch*0.2:
                # if FLAGS.eval:
                #     save_img = np.hstack((test_input[0,:,:,0],result[4][0],result[3][0]))
                #     save_img = np.uint8(save_img*255)
                #     cv2.imwrite(f'./result/{it}.jpg',save_img)

                # y_true, y_pred = result[3], result[4]
                # test_masked_dice += masked_np_dice(y_true=y_true, y_pred=y_pred,mask=test_mask)

                test_count += 1
                this_class = np.argmax(test_label, axis=-1)[0]
                # class_accs[this_class] += result[1] # added for debug
                # class_samples[this_class] += 1
            test_acc = test_acc/test_count
            # test_dice = test_dice/test_count

            # test_masked_dice = test_dice
            # y_true = y_true[:int(len(y_true)*0.2)]
            # y_pred = y_pred[:int(len(y_pred)*0.2)]
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            x_input = np.concatenate(x_input)
            y_pred_post = np.zeros_like(y_pred)

            case_index.append(len(y_true))
            for k in range(len(case_index)-1):
                y_pred_post[case_index[k]:case_index[k+1]] = post(y_pred[case_index[k]:case_index[k+1]],min_size=20)
                # sitk.WriteImage(sitk.GetImageFromArray(y_pred[case_index[k]:case_index[k+1]]),f'./nii/pred{k}.nii')
                # sitk.WriteImage(sitk.GetImageFromArray(y_pred_post[case_index[k]:case_index[k+1]]),f'./nii/pred_post{k}.nii')
                # sitk.WriteImage(sitk.GetImageFromArray(y_true[case_index[k]:case_index[k+1]]),f'./nii/gt{k}.nii')
                # print(k,y_pred[case_index[k]:case_index[k+1]].sum()-y_pred_post[case_index[k]:case_index[k+1]].sum())

            # y_true2 = np.zeros((len(y_true),512,512))
            # y_pred2 = np.zeros((len(y_true),512,512))
            # y_pred_post2 = np.zeros((len(y_true),512,512))
            # for k in range(len(y_true)):
            #     y_true2[k] = cv2.resize(y_true[k].astype(np.uint8),(512,512),cv2.INTER_NEAREST)
            #     y_pred2[k] = cv2.resize(y_pred[k].astype(np.uint8),(512,512),cv2.INTER_NEAREST)
            #     y_pred_post2[k] = cv2.resize(y_pred_post[k].astype(np.uint8),(512,512),cv2.INTER_NEAREST)

            test_dice = np_dice(y_true=y_true, y_pred=y_pred)
            test_masked_dice = np_dice(y_true=y_true, y_pred=y_pred_post)

            # if FLAGS.eval:
            #     for k in range(len(y_true)):
            #         save_img = np.hstack((x_input[k],y_pred[k],y_pred_post[k],y_true[k]))
            #         save_img = np.uint8(save_img*255)
            #         cv2.imwrite(f'./result/{k}.jpg',save_img)

            # y_true2 = y_true2[:int(len(y_true2)*0.2)]
            # y_pred2 = y_pred2[:int(len(y_pred2)*0.2)]
            # y_true2 = np.concatenate(y_true2)
            # y_pred2 = np.concatenate(y_pred2)
            # test_masked_dice = np_dice(y_true=y_true2, y_pred=y_pred2)

            if not FLAGS.eval:
            # test_masked_dice = test_masked_dice/test_count
            # if test_acc > best_test_acc:
            #     best_test_acc = test_acc
            #     saver.save(sess, FLAGS.logdir + '/' + exp_string + '/itr' + str(itr) + '_model_acc' + str(best_test_acc))
                if test_masked_dice > best_test_masked_dice:
                    best_test_masked_dice = test_masked_dice
                    best_test_step = itr
                    if FLAGS.suffix:
                        saver.save(sess, FLAGS.logdir + '/' + exp_string + f'/model_{FLAGS.suffix}_itr' + str(itr) + f'_dice{best_test_masked_dice:.5f}',latest_filename=f'checkpoint_{FLAGS.suffix}')
                    else:
                        saver.save(sess, FLAGS.logdir + '/' + exp_string + f'/model_itr' + str(itr) + f'_dice{best_test_masked_dice:.5f}')

            print('Unseen Target Validation results: Iteration %d, Loss: %.5f, Accuracy: %.5f Dice: (%.5f,%.5f)' %(itr, test_loss/test_count, test_acc,test_dice,test_masked_dice))
            if FLAGS.eval:
                break
            print('Current best dice {:.5f}, best step {} '.format(best_test_masked_dice,best_test_step))   

            if FLAGS.suffix:
                eval_file = os.path.join(FLAGS.logdir,exp_string,f'eval_{FLAGS.suffix}.txt')
            else:
                eval_file = os.path.join(FLAGS.logdir,exp_string,'eval.txt')  

            with open(eval_file, 'a') as fle:
                fle.write('Unseen Target Validation results: Iteration %d, Loss: %.5f, Accuracy: %.5f, Dice: (%.5f,%.5f) \n' %(itr, test_loss/test_count, test_acc, test_dice,test_masked_dice))

def main():

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    filelist_root = './LIVER/'  # path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line
    source_list = ['BTCV', 'CHAOS', 'LITS', 'IRCAD']
    source_list.remove(FLAGS.target_domain)

    exp_string = 'mame_'  + FLAGS.target_domain + os.sep #'_' + FLAGS.loss + '_' + FLAGS.target_domain + os.sep 
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
    model = MASF(task='seg')
    model.construct_model_train()
    model.construct_model_test()
    
    model.summ_op = tf.summary.merge_all()
    sess = tf.InteractiveSession()
    
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var_to_restore = [val for val in var if 'output' not in str(val) and 'fc9' not in str(val)]
    saver = tf.train.Saver(var_to_restore)
    #########warmup#########
    model_file = glob(f'./pretrained_weights/unet/{FLAGS.target_domain}/*.index')[0].replace('.index','')
    reader = pywrap_tensorflow.NewCheckpointReader(model_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)
    #     print(reader.get_tensor(key))
    warmup_itr = int(model_file.split('itr')[-1].split('_')[0])
    warmup_dice = float(model_file.split('dice')[-1])
    print(f"Loading warmup model weights from {model_file}, warmup_itr is {warmup_itr}, warmup_dice is {warmup_dice:.5f}")
    saver.restore(sess, model_file)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=3)
    output_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(output_var)

    # print('Loading pretrained weights')
    # model.load_initial_weights(sess)
    # print('Training Unet from scratch')
    tf.get_default_graph()


    resume_itr = 0
    #checkpoint_dir = ''
    checkpoint_dir = './log/mame_BTCV_L_all_BTCV/deepall_mbs_120.inner5e-05.outer5e-05.clipNorm2.0.metric1e-05.margin20.0_flip(ud+lr)_128/'
    if FLAGS.resume or FLAGS.eval:
        model_file = tf.train.latest_checkpoint(checkpoint_dir)
        if model_file:
            # ind1 = model_file.index('model')
            # resume_itr = int(model_file[ind1+5:])
            resume_itr = int(model_file.split('itr')[-1].split('_')[0])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    train_file_list = [os.path.join(filelist_root, source_domain+'_list.txt') for source_domain in source_list]
    # test_file_list = [os.path.join(filelist_root, FLAGS.target_domain+'_list.txt')]
    test_file_list = [os.path.join(filelist_root, FLAGS.target_domain+'_list_test.txt')]
    train(model, saver, sess, exp_string, train_file_list, test_file_list, resume_itr)

if __name__ == "__main__":
    main()

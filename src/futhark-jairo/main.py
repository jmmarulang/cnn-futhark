#!/usr/bin/env python3

from mnist import MNIST
import argparse
import futhark_server
import numpy as np
import time
import logging

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger('futhark')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Futhark implementation')
    parser.add_argument('mnistpath', metavar='MNIST_PATH', type=str,
                        help='path to where mnist data is located.')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='batch size')
    parser.add_argument('-t', '--training-size', type=int, default=1,
                        help='training size (max 60000)')
    parser.add_argument('-p', '--epoch', type=int, default=5,
                        help='epochs to iterate')
    parser.add_argument('-r', '--training-rate', type=float, default=0.05,
                        help='rate to learn at')
    parser.add_argument('--futhark', type=str, help='server-mode binary')

    args = parser.parse_args()

    # Get data
    mndata = MNIST(args.mnistpath, return_type='numpy')

    # Process data
    train_images_raw, train_labels_raw = mndata.load_training()
    test_images_raw, test_labels_raw = mndata.load_testing()
    train_images = train_images_raw.astype(np.float32, copy=False).reshape(train_images_raw.shape[0], 28, 28)
    train_labels = train_labels_raw.astype(np.int8, copy=False)

    logger.info(f'Parameters: epoch({args.epoch}), batch-size({args.batch_size}), training-rate({args.training_rate}), training-size({args.training_size})')
    print(args.futhark)
    with futhark_server.Server(args.futhark) as server:
        server.put_value('batchsize', np.int64(args.batch_size))
        server.put_value('rate', np.float32(args.training_rate))
        server.put_value('trainings', np.int64(args.training_size))
        server.put_value('imgs', train_images)
        server.put_value('lbls', train_labels)
        server.cmd_call('initial_state', 'state')
        train_start = time.perf_counter ()
        for epoch in range(args.epoch):
            server.cmd_call('iteration',
                            *['state_new_and_error'],
                            *['trainings', 'batchsize', 'rate', 'imgs', 'lbls', 'state'])
            state_new_and_error = server.get_value('state_new_and_error')
            server.put_value('b', state_new_and_error[0])
            server.put_value('b1', state_new_and_error[1])
            server.put_value('b2', state_new_and_error[2])
            server.put_value('fc', state_new_and_error[3])
            server.put_value('k1', state_new_and_error[4])
            server.put_value('k2', state_new_and_error[5])
            server.cmd_free('state_new_and_error', 'state')
            server.cmd_call('make_state', 'state', 'k1', 'b1', 'k2', 'b2','fc','b')
            error =  state_new_and_error[6]
            logger.info ('Epoch {}, Loss: {:f}'.format (epoch+1, error))
            server.cmd_free('k1', 'b1', 'k2', 'b2','fc','b')
        train_stop = time.perf_counter () - train_start
        logger.info('Train: {:f}s'.format(train_stop))

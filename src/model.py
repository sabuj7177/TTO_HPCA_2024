import logging

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, args):
        self.args = args
        if args.synthetic_data_size:
            self.size = args.synthetic_data_size
            assert args.only_allreduce or args.only_reduce_scatter or args.only_all_gather
        else:
            self.size = 0
            self.num_layers = 0
            self.layers = []
            self.name = args.network.split('/')[-1].split('.')[0]

            self.parse_model()

    def parse_model(self):
        param_file = open(self.args.network, 'r')
        layer_info = []
        prepend = ""

        first = True
        current_layer = 0
        max_layer = 0

        logger.info('\nModel loading ...')
        for row in param_file:
            if first:
                first = False
                continue

            elems = row.strip().split(',')

            # Do not continue if incomplete line
            if len(elems) < 9:
                if elems[0]:
                    logger.warn('Warn: incomplete model layer description: line {}'.format(self.num_layers + 1))
                    logger.warn(' -- {}'.format(row))
                continue

            if self.args.layer_by_layer:
                if current_layer not in self.args.layer_number_list:
                    current_layer += 1
                    continue
            self.layers.append({})
            self.layers[self.num_layers]['name'] = elems[0]
            self.layers[self.num_layers]['ifmap_h'] = int(elems[1])
            self.layers[self.num_layers]['ifmap_w'] = int(elems[2])
            self.layers[self.num_layers]['filter_h'] = int(elems[3])
            self.layers[self.num_layers]['filter_w'] = int(elems[4])
            self.layers[self.num_layers]['num_channels'] = int(elems[5])
            self.layers[self.num_layers]['num_filters'] = int(elems[6])
            self.layers[self.num_layers]['stride'] = int(elems[7])

            layer_size = int(elems[3]) * int(elems[4]) * int(elems[5]) * int(elems[6])
            self.layers[self.num_layers]['size'] = layer_size
            self.num_layers += 1

            self.size += layer_size
            # print("Layer size " + str(layer_size))
            if layer_size < 32768:
                prepend += str(current_layer) + "_"
            else:
                layer_info.append(prepend + str(current_layer))
                prepend = ""
            current_layer += 1
            if layer_size > max_layer:
                max_layer = layer_size

        # print(layer_info)
        logger.info('Model loading finished\n')
        print("Max layer size " + str(max_layer))
        for l in range(self.num_layers):
            logger.debug('layer: {}: [name: {}, ifmap_h: {}, ifmap_w: {},'
                         'filter_h: {}, filter_w: {}, num_channels: {},'
                         'num_filters: {}, stride: {}]'.format(l,
                             self.layers[l]['name'], self.layers[l]['ifmap_h'],
                             self.layers[l]['ifmap_w'],
                             self.layers[l]['filter_h'],
                             self.layers[l]['filter_w'],
                             self.layers[l]['num_channels'],
                             self.layers[l]['num_filters'],
                             self.layers[l]['stride']))
    # parse_model() end

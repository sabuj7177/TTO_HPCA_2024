import copy
import sys
import math
import numpy as np
from copy import deepcopy
import logging

sys.path.append('booksim2/src')

from sim_object import SimObject
import pybooksim
from npu import NPU
from eventq import EventQueue
from message_buffer import *

logger = logging.getLogger(__name__)

class HMC(SimObject):
    # like static variables, reducing simulation time for data-parallel training
    inference_cycles = None
    training_cycles = None
    model_aggregation_cycles = None
    allreduce_aggregation_cycles = {}
    cur_mid = 0 # global message ID
    # XXX: oracle lockstep
    allreduce_timestep = None
    allreduce_remaining_for_timestep = None
    hmcs = []
    back_time_temp = None

    def __init__(self, i, args, eventq):
        super().__init__(eventq)
        HMC.hmcs.append(self)

        self.id = i
        self.name = 'HMC-{}'.format(self.id)
        self.args = args
        self.npu = NPU(args)
        self.num_npus = self.args.num_vaults

        self.local_eventq = EventQueue()
        self.computation_state = None
        self.communication_state = None

        self.compute_cycles = 0
        self.allreduce_compute_cycles = 0

        self.from_network_message_buffers = None
        self.to_network_message_buffers = None

        self.model = None
        self.bytes_per_param = 4 # bytes
        self.samples_per_npu = math.ceil(self.args.mini_batch_size / (self.args.num_vaults * self.args.num_hmcs))

        self.message_size = args.message_size # bytes
        self.sub_message_size = args.sub_message_size
        self.base_num_messages = None
        self.base_num_sub_messages = None
        self.num_messages = [None for i in range(self.args.radix)]
        self.num_sub_messages = [None for i in range(self.args.radix)]

        # for the schedule semantics, refer to allreduce/allreduce.py
        self.allreduce = None
        self.reduce_scatter_schedule = None
        self.all_gather_schedule = None
        self.new_step = True
        self.estimated_steptime = None
        self.estimated_next_steptime = None

        self.cur_mids = np.zeros(self.args.radix, dtype=int)
        for i in range(self.args.radix):
            self.cur_mids[i] = HMC.cur_mid
            HMC.cur_mid = (HMC.cur_mid + 1) % 2147483647
        self.sending = None
        self.free_nis = set([i for i in range(self.args.radix)])
        self.just_allocated_nis = {}
        self.pending_aggregations = []
        self.total_messages_sent = 0
        self.link_dict = None
        self.back_time = None


    '''
    load_model() - assign the NN model to this hmc
    @model: the NN model to be loaded
    '''
    def load_model(self, model):
        self.model = model
        if self.message_size == 0:
            self.base_num_messages = 1
            self.base_num_sub_messages = math.ceil(self.model.size * self.bytes_per_param /
                    self.sub_message_size / self.args.num_hmcs)
            if self.base_num_sub_messages <= 58:
                self.estimated_steptime = self.base_num_sub_messages
            else:
                self.estimated_steptime = self.base_num_sub_messages * 16 - 58 * 16
        else:
            assert self.message_size >= self.sub_message_size
            self.base_num_messages = math.ceil(self.model.size * self.bytes_per_param /
                    self.message_size / self.args.num_hmcs)
            self.base_num_sub_messages = math.ceil(self.message_size / self.sub_message_size)
            if self.base_num_messages <= 58:
                self.estimated_steptime = self.base_num_messages
            else:
                self.estimated_steptime = self.base_num_messages * 17 - 58 * 16
    # end of load_model()


    '''
    startup() - startup function for simulation of HMC

    desc - schedule the start event for the simulation of HMC. Currently, assuming
           we are doing training only.
    TODO: should be extended later for more functionalities
    '''
    def startup(self):
        # currently start from training
        self.communication_state = 'idle'
        if self.args.only_allreduce or self.args.only_reduce_scatter:
            self.computation_state = 'aggregating'
            self.local_eventq.schedule('finish-aggregation', 0)
        elif self.args.only_all_gather:
            self.computation_state = 'idle'
            self.communication_state = 'all-gather'
            self.local_eventq.schedule('all-gather', 0)
        else: # need to do training, including only-compute
            self.computation_state = 'idle'
            self.local_eventq.schedule('training', 0)
        self.global_eventq.schedule(self, 0)
    # end of startup()


    '''
    set_allreduce() - set allreduce schedule
    @allreduce: allreduce schedule
    '''
    def set_allreduce(self, allreduce):
        if self.args.oracle_lockstep:
            if HMC.allreduce_timestep == None:
                HMC.allreduce_timestep = 0
                HMC.allreduce_remaining_for_timestep = [0] * (allreduce.timesteps * 2 + 1)
                if self.args.only_all_gather:
                    HMC.allreduce_timestep = allreduce.timesteps + 1

        self.allreduce = allreduce
        self.reduce_scatter_schedule = deepcopy(allreduce.reduce_scatter_schedule[self.id])
        self.all_gather_schedule = deepcopy(allreduce.all_gather_schedule[self.id])
        self.messages_received = {'reduce-scatter': [{} for i in range(self.args.num_hmcs)],
                                  'all-gather': [0] * self.args.num_hmcs}
        self.messages_received_end = {'reduce-scatter': [{} for i in range(self.args.num_hmcs)],
                                      'all-gather': [False] * self.args.num_hmcs}
        self.messages_sent = allreduce.messages_sent
        self.link_dict = allreduce.link_dict
        self.sending = allreduce.sending

        for flow in range(self.args.num_hmcs):
            for child in range(self.args.num_hmcs):
                self.messages_received['reduce-scatter'][flow][child] = 0
                self.messages_received_end['reduce-scatter'][flow][child] = False
    # end of set_allreduce()


    '''
    set_message_buffers() - set message buffers connected with network
    @from_network_message_buffers: message buffers for incoming messages
    @to_network_message_buffers: message buffers for outgoing messages
    '''
    def set_message_buffers(self, from_network_message_buffers, to_network_message_buffers):
        assert len(from_network_message_buffers) == self.args.radix
        assert len(to_network_message_buffers) == self.args.radix
        self.from_network_message_buffers = from_network_message_buffers
        self.to_network_message_buffers = to_network_message_buffers
    # end of set_message_buffers


    '''
    schedule() - schedule the event at a given time
    @event: the event to be scheduled
    @cycle: scheduled time
    '''
    def schedule(self, event, cycle):
        self.local_eventq.schedule(event, cycle)
        self.global_eventq.schedule(self, cycle)
    # end of schedule()


    '''
    reschedule() - reschedule the event due to structure hazard
    @event: the event to be rescheduled
    '''
    def reschedule(self, event):
        next_cycle = self.local_eventq.next_event_cycle()
        self.local_eventq.schedule(event, next_cycle)
    # end of reschedule()


    '''
    process() - event processing function in a particular cycle
    @cur_cycle: the current cycle that with events to be processed
    '''
    def process(self, cur_cycle):
        events = self.local_eventq.get_events(cur_cycle)
        # Evaluate the events
        for event in events:
            if event == 'training':
                self.training_evaluate(cur_cycle)
            elif event == 'finish-training':
                self.finish_training_evaluate(cur_cycle)
            elif event == 'aggregation':
                self.aggregation_evaluate(cur_cycle)
            elif event == 'finish-aggregation':
                self.finish_aggregation_evaluate(cur_cycle)
            elif event == 'reduce-scatter':
                self.reduce_scatter_evaluate_optimal(cur_cycle)
            elif event == 'send-reduce-message':
                self.send_reduce_message_evaluate_optimal(cur_cycle)
            elif event == 'all-gather':
                self.all_gather_evaluate_optimal(cur_cycle)
            elif event == 'send-gather-message':
                self.send_gather_message_evaluate_optimal(cur_cycle)
            elif event == 'incoming-message':
                self.incoming_message_evaluate_optimal(cur_cycle)
            else:
                raise RuntimeError('Unknown event type {} for {}'.format(event, self.name))

        # Update the states according to the events
        for event in events:
            if event == 'training':
                self.training_update(cur_cycle)
            elif event == 'finish-training':
                self.finish_training_update(cur_cycle)
            elif event == 'aggregation':
                self.aggregation_update(cur_cycle)
            elif event == 'finish-aggregation':
                self.finish_aggregation_update(cur_cycle)
            elif event == 'reduce-scatter':
                self.reduce_scatter_update(cur_cycle)
            elif event == 'send-reduce-message':
                self.send_reduce_message_update(cur_cycle)
            elif event == 'all-gather':
                self.all_gather_update(cur_cycle)
            elif event == 'send-gather-message':
                self.send_gather_message_update(cur_cycle)
            else:
                assert event == 'incoming-message'
                self.incoming_message_update(cur_cycle)
    # end of process()


    '''
    training_evaluate() - change to transient state
    '''
    def training_evaluate(self, cur_cycle):
        assert self.computation_state == 'idle'
        self.computation_state = 'idle-to-training'
    # end of training_evaluate()

    '''
    training_update() - update the state for training action
    '''
    def training_update(self, cur_cycle):
        assert self.computation_state == 'idle-to-training'
        self.computation_state = 'training'
        cycles = self.train()
        self.schedule('finish-training', cur_cycle + cycles)
        logger.info('{} | {} | starts training, computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
    # end of training_update()


    '''
    finish_training_evaluate() - change to transient state
    '''
    def finish_training_evaluate(self, cur_cycle):
        assert self.computation_state == 'training'
        self.computation_state = 'training-to-idle'
    # end of finish_training_evaluate()

    '''
    finish_training_update() - update the state and schedule dependent events
    '''
    def finish_training_update(self, cur_cycle):
        assert self.computation_state == 'training-to-idle'
        self.computation_state = 'idle'
        self.schedule('aggregation', cur_cycle + 1)
        logger.info('{} | {} finishes training, computation sate: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
    # end of finish_training_update()


    '''
    aggregation_evaluate() - change to transient state
    '''
    def aggregation_evaluate(self, cur_cycle):
        if self.computation_state == 'idle':
            self.computation_state = 'idle-to-aggregating'
    # end of aggregation_evaluate()

    '''
    aggregation_update() - action execution
    '''
    def aggregation_update(self, cur_cycle):
        if self.computation_state == 'idle-to-aggregating':
            self.computation_state = 'aggregating'
            cycles = self.aggregate()
            self.schedule('finish-aggregation', cur_cycle + cycles)
            logger.info('{} | {} | starts aggregation, computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
        else:
            self.reschedule('aggregation')
            logger.debug('{} | {} | compute is not available for aggregation, state {}'.format(cur_cycle, self.name, self.computation_state))
    # end of aggregation_update()


    '''
    finish_aggregation_evaluate() - change to transient state
    '''
    def finish_aggregation_evaluate(self, cur_cycle):
        assert self.computation_state == 'aggregating'
        self.computation_state = 'aggregating-to-idle'
    # end of finish_aggregation_evalaute()

    '''
    finish_aggregation_update() - update the state and schedule dependent events
    '''
    def finish_aggregation_update(self, cur_cycle):
        assert self.computation_state == 'aggregating-to-idle'
        self.computation_state = 'idle'
        # aggregate_end = True

        if self.communication_state == 'idle': # local aggregation
            if self.args.only_compute:
                return
            assert len(self.pending_aggregations) == 0
            self.communication_state = 'reduce-scatter'
            assert len(self.reduce_scatter_schedule) > 0
            self.schedule('reduce-scatter', cur_cycle + 1)

        elif len(self.pending_aggregations) > 0: # allreduce aggregation
            flow, child, num_message, chunk_id = self.pending_aggregations.pop(0)
            logger.info('{} | {} | clear pending aggregation for flow {} from child HMC-{} for chunk {}'.format(cur_cycle, self.name, flow, child, chunk_id))
            dep_resolved = False
            for key in self.reduce_scatter_schedule.keys():
                for schedule in self.reduce_scatter_schedule[key]:
                    if schedule[0] == flow and schedule[1] == chunk_id and child in schedule[2]:
                        schedule[2].remove(child)
                        dep_resolved = True
                        break
            if self.args.allreduce == 'mesh_fermat':
                dep_to_delete_first = None
                if not dep_resolved:
                    for dep_info in self.allreduce.rs1_final_dep[self.id]:
                        if flow == self.id and dep_info[0] == chunk_id and child in dep_info[1]:
                            dep_info[1].remove(child)
                            dep_resolved = True
                            if len(dep_info[1]) == 0:
                                dep_to_delete_first = dep_info
                            break
                if dep_to_delete_first is not None:
                    self.allreduce.rs1_final_dep[self.id].remove(dep_to_delete_first)
            dep_to_delete_second = None
            if not dep_resolved:
                for dep_info in self.allreduce.rs2_final_dep[self.id]:
                    if flow == self.id and dep_info[0] == chunk_id and child in dep_info[1]:
                        dep_info[1].remove(child)
                        dep_resolved = True
                        if len(dep_info[1]) == 0:
                            dep_to_delete_second = dep_info
                        break
            if dep_to_delete_second is not None:
                self.allreduce.rs2_final_dep[self.id].remove(dep_to_delete_second)
            if not dep_resolved:
                raise RuntimeError("Dependency is not resolved!!!")
            self.schedule('reduce-scatter', cur_cycle + 1)
        logger.info('{} | {} | finishes aggregation , computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))

        if len(self.pending_aggregations) > 0:
            self.schedule('aggregation', cur_cycle + 1)
    # end of finish_aggregation_update()

    # Specific reduce scatter function for heterogeneous optimal with strict ordering
    def reduce_scatter_evaluate_optimal(self, cur_cycle):
        all_links_empty = True
        for key in self.link_dict[self.id].keys():
            if self.link_dict[self.id][key]:
                all_links_empty = False
                break
        all_rs_done = True
        for key in self.link_dict[self.id].keys():
            if len(self.reduce_scatter_schedule[key]) > 0:
                all_rs_done = False
                break
        all_fd_final_dep_resolved = True
        if self.args.allreduce == 'mesh_fermat':
            for key in self.allreduce.rs1_final_dep.keys():
                if len(self.allreduce.rs1_final_dep[key]) > 0:
                    all_fd_final_dep_resolved = False
                    break
        all_sd_final_dep_resolved = True
        for key in self.allreduce.rs2_final_dep.keys():
            if len(self.allreduce.rs2_final_dep[key]) > 0:
                all_sd_final_dep_resolved = False
                break
        if all_links_empty and all_rs_done and all_sd_final_dep_resolved:
            self.communication_state = 'all-gather'
            self.schedule('all-gather', cur_cycle + 1)
            return

        for key in self.link_dict[self.id].keys():
            if not self.link_dict[self.id][key] and len(self.reduce_scatter_schedule[key]) > 0:
                schedule = self.reduce_scatter_schedule[key][0]
                source = self.id
                dest = key
                tree_id = schedule[0]
                chunk_id = schedule[1]
                dependencies = schedule[2]
                total_messages = schedule[3]
                order = schedule[4]
                source_ni = schedule[5]
                dest_ni = schedule[6]
                if order == 1 and not all_fd_final_dep_resolved:
                    continue
                if len(dependencies) > 0:
                    continue
                assert self.sending[self.id][source_ni] is None
                assert self.messages_sent[self.id][source_ni] == 0
                self.sending[self.id][source_ni] = (tree_id, source, dest, source_ni, dest_ni, chunk_id)
                self.link_dict[self.id][key] = True
                self.reduce_scatter_schedule[key].pop(0)
                self.num_messages[source_ni] = total_messages
                self.num_sub_messages[source_ni] = self.base_num_sub_messages
                logger.info(
                    '{} | {} | start reducing for flow {} (from NI {}) to parent HMC-{} (to NI {}) for chunk {}'.format(
                        cur_cycle, self.name, tree_id, source_ni, dest, dest_ni, chunk_id))
                self.schedule("send-reduce-message", cur_cycle + 1)
        if self.communication_state == 'reduce-scatter':
            self.schedule('reduce-scatter', cur_cycle + 1)

    # Specific send reduce message function for heterogeneous optimal with strict ordering
    def send_reduce_message_evaluate_optimal(self, cur_cycle):
        for ni, data in enumerate(self.sending[self.id]):
            if data == None:
                continue
            if self.messages_sent[self.id][ni] >= self.num_messages[ni] * self.num_sub_messages[ni]:
                continue
            if not self.to_network_message_buffers[ni].is_full():
                flow = data[0]
                src = data[1]
                dest = data[2]
                src_ni = data[3]
                dest_ni = data[4]
                chunk_id = data[5]

                if flow == src or flow == self.id:
                    continue

                src_node = self.id * self.args.radix + ni
                dest_node = dest * self.args.radix + dest_ni
                submsgtype = pybooksim.Message.HeadTail
                self.messages_sent[self.id][ni] += 1
                self.total_messages_sent += 1
                end = False
                if self.messages_sent[self.id][ni] == self.num_messages[ni] * self.num_sub_messages[ni]:
                    end = True
                message = Message(flow, self.cur_mids[ni], src_node, dest_node, self.sub_message_size, pybooksim.Message.ReduceData, submsgtype, chunk_id, end)
                self.cur_mids[ni] = HMC.cur_mid
                HMC.cur_mid = (HMC.cur_mid + 1) % 2147483647
                self.to_network_message_buffers[ni].enqueue(message, cur_cycle, 1)
            if self.messages_sent[self.id][ni] < self.num_messages[ni] * self.num_sub_messages[ni]:
                self.schedule('send-reduce-message', cur_cycle + 1)

    # Specific all gathering for heterogeneous optimal with strict ordering
    def all_gather_evaluate_optimal(self, cur_cycle):
        if self.args.only_reduce_scatter:
            return
        all_links_empty = True
        for key in self.link_dict[self.id].keys():
            if self.link_dict[self.id][key]:
                all_links_empty = False
                break
        all_ag_done = True
        for key in self.link_dict[self.id].keys():
            if len(self.all_gather_schedule[key]) > 0:
                all_ag_done = False
                break
        all_fd_final_dep_resolved = True
        if self.args.allreduce == 'mesh_fermat':
            for key in self.link_dict[self.id].keys():
                for schedule in self.all_gather_schedule[key]:
                    if schedule[4] == 0:
                        all_fd_final_dep_resolved = False
                        break
        if all_links_empty and all_ag_done:
            self.communication_state = 'idle'
            return

        for key in self.link_dict[self.id].keys():
            if not self.link_dict[self.id][key] and len(self.all_gather_schedule[key]) > 0:
                schedule = self.all_gather_schedule[key][0]
                source = self.id
                dest = key
                tree_id = schedule[0]
                chunk_id = schedule[1]
                dependencies = schedule[2]
                total_messages = schedule[3]
                order = schedule[4]
                source_ni = schedule[5]
                dest_ni = schedule[6]
                if order == 1 and not all_fd_final_dep_resolved:
                    continue
                if len(dependencies) > 0:
                    continue
                assert self.sending[self.id][source_ni] is None
                assert self.messages_sent[self.id][source_ni] == 0
                self.sending[self.id][source_ni] = (tree_id, source, dest, source_ni, dest_ni, chunk_id)
                self.link_dict[self.id][key] = True
                self.all_gather_schedule[key].pop(0)
                self.num_messages[source_ni] = total_messages
                self.num_sub_messages[source_ni] = self.base_num_sub_messages
                logger.info(
                    '{} | {} | start gathering for flow {} (from NI {}) to parent HMC-{} (to NI {}) for chunk {}'.format(
                        cur_cycle, self.name, tree_id, source_ni, dest, dest_ni, chunk_id))
                self.schedule("send-gather-message", cur_cycle + 1)
        if self.communication_state == 'all-gather':
            self.schedule('all-gather', cur_cycle + 1)

    # Specific send gather function for heterogeneous optimal with strict ordering
    def send_gather_message_evaluate_optimal(self, cur_cycle):
        if self.args.only_reduce_scatter:
            return
        for ni, data in enumerate(self.sending[self.id]):
            if data == None:
                continue
            if self.messages_sent[self.id][ni] >= self.num_messages[ni] * self.num_sub_messages[ni]:
                continue
            if not self.to_network_message_buffers[ni].is_full():
                flow = data[0]
                src = data[1]
                dest = data[2]
                src_ni = data[3]
                dest_ni = data[4]
                chunk_id = data[5]

                src_node = self.id * self.args.radix + ni
                dest_node = dest * self.args.radix + dest_ni
                submsgtype = pybooksim.Message.HeadTail
                self.messages_sent[self.id][ni] += 1
                self.total_messages_sent += 1
                end = False
                if self.messages_sent[self.id][ni] == self.num_messages[ni] * self.num_sub_messages[ni]:
                    end = True
                message = Message(flow, self.cur_mids[ni], src_node, dest_node, self.sub_message_size, pybooksim.Message.GatherData, submsgtype, chunk_id, end)
                self.cur_mids[ni] = HMC.cur_mid
                HMC.cur_mid = (HMC.cur_mid + 1) % 2147483647
                self.to_network_message_buffers[ni].enqueue(message, cur_cycle, 1)
            if self.messages_sent[self.id][ni] < self.num_messages[ni] * self.num_sub_messages[ni]:
                self.schedule('send-gather-message', cur_cycle + 1)

    # Specific incoming message function for heterogeneous optimal with strict ordering
    def incoming_message_evaluate_optimal(self, cur_cycle):
        for ni, message_buffer in enumerate(self.from_network_message_buffers):
            message = message_buffer.peek(cur_cycle)
            if message == None:
                continue
            self.from_network_message_buffers[ni].dequeue(cur_cycle)
            src = message.src // self.args.radix
            src_ni = message.src % self.args.radix
            if message.type == pybooksim.Message.ReduceData:
                self.messages_received['reduce-scatter'][message.flow][src] += 1
                if message.end:
                    logger.info(
                        '{} | {} | receives full reduce for flow {} from child HMC-{} for chunk {}'.format(cur_cycle, self.name,
                                                                                              message.flow, src, message.priority))
                    # if src == 5:
                    # print('{} | {} | receives full reduce for flow {} from child HMC-{} for chunk {}'.format(cur_cycle, self.name,
                    #                                                                           message.flow, src, message.priority))
                    self.link_dict[src][self.id] = False
                    self.sending[src][src_ni] = None
                    self.messages_sent[src][src_ni] = 0
                    self.schedule('reduce-scatter', cur_cycle + 1)
                    if self.computation_state == 'idle':
                        self.schedule('aggregation', cur_cycle + 1)
                    self.pending_aggregations.append(
                            (message.flow, src, self.messages_received['reduce-scatter'][message.flow][src], message.priority))
                    self.messages_received['reduce-scatter'][message.flow][src] = 0
            elif message.type == pybooksim.Message.GatherData:
                self.messages_received['all-gather'][message.flow] += 1
                if message.end:
                    logger.info(
                        '{} | {} | receives full gather for flow {} from parent HMC-{} for chunk {}'.format(cur_cycle, self.name,
                                                                                               message.flow, src, message.priority))
                    # print('{} | {} | receives full gather for flow {} from parent HMC-{} for chunk {}'.format(cur_cycle, self.name,
                    #                                                                            message.flow, src, message.priority))
                    self.link_dict[src][self.id] = False
                    self.sending[src][src_ni] = None
                    self.messages_sent[src][src_ni] = 0
                    self.schedule('all-gather', cur_cycle + 1)
                    for key in self.all_gather_schedule.keys():
                        for schedule in self.all_gather_schedule[key]:
                            if schedule[0] == message.flow and schedule[1] == message.priority and src in schedule[2]:
                                schedule[2].remove(src)
                                break


    '''
    reduce_scatter_update() - schedule selected communications
    '''
    def reduce_scatter_update(self, cur_cycle):
        return
    # end of reduce_scatter_update()


    '''
    send_reduce_message_update() - try to schedule event to select remaining communications
    '''
    def send_reduce_message_update(self, cur_cycle):
        return
    # # end of send_reduce_message_update()


    '''
    all_gather_update() - schedule selected communications
    '''
    def all_gather_update(self, cur_cycle):
        return
    # end of all_gather_update()


    '''
    send_gather_message_update() - try to schedule event to select remaining communications
    '''
    def send_gather_message_update(self, cur_cycle):
        return
    # end of send_gather_message_update()


    '''
    incoming_message_update() - check states and try to schedule event to select remaining communications
    '''
    def incoming_message_update(self, cur_cycle):
        return
    # end of incoming_message_update()


    '''
    aggregate() - aggregate all the weight updates of all the NPUs

    return: the cycles of aggregation
    '''
    def aggregate(self):
        if not self.pending_aggregations:
            if HMC.model_aggregation_cycles == None:
                partial_model_per_npu = math.ceil(self.model.size / self.num_npus)
                cycles = self.npu.aggregate(partial_model_per_npu, self.num_npus)
                HMC.model_aggregation_cycles = cycles

            self.compute_cycles += HMC.model_aggregation_cycles

            return HMC.model_aggregation_cycles

        else:
            flow, child, num_sub_messages, chunk_id = self.pending_aggregations[0]
            data_size = num_sub_messages * self.sub_message_size // self.bytes_per_param  # NO. params
            if data_size not in HMC.allreduce_aggregation_cycles.keys():
                partial_aggregation_per_npu = math.ceil(data_size / self.num_npus)
                cycles = self.npu.aggregate(partial_aggregation_per_npu, self.num_npus)
                HMC.allreduce_aggregation_cycles[data_size] = cycles

            self.allreduce_compute_cycles += HMC.allreduce_aggregation_cycles[data_size]

            return HMC.allreduce_aggregation_cycles[data_size]
    # end of aggregate()


    '''
    inference() - inference processing of the NN model

    return: number of cycles for parallel inference
    '''
    def inference(self):

        if HMC.inference_cycles == None:
            npu_cycles = self.npu.inference(self.model)
            cycles = npu_cycles * self.samples_per_npu
            HMC.inference_cycles = cycles

        self.compute_cycles += HMC.inference_cycles

        return HMC.inference_cycles
    # end of inference()


    '''
    train() - training of the NN model

    return: number of cycles for training
    '''
    def train(self):

        if HMC.training_cycles == None:
            npu_cycles = self.npu.train(self.model)
            cycles = npu_cycles * self.samples_per_npu
            HMC.training_cycles = cycles
            HMC.back_time_temp = self.npu.back_time

        self.compute_cycles += HMC.training_cycles
        self.back_time = HMC.back_time_temp

        return HMC.training_cycles
    # end of train()


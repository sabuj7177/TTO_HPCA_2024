import sys

sys.path.append('booksim2/src')

from sim_object import SimObject
import pybooksim
from eventq import EventQueue
from message_buffer import *


class BookSim(SimObject):
    def __init__(self, args, eventq):
        super().__init__(eventq)
        self.name = 'BookSimNI'
        self.args = args
        self.booksim = pybooksim.BookSim(args.booksim_config)
        self.local_eventq = EventQueue()
        self.in_message_buffers = None
        self.out_message_buffers = None
        self.reduce_scatter_time_track_dict = None
        self.all_gather_time_track_dict = None
        self.per_message_max_latency = None
        self.links_usage = None
        self.link_start_times = {}
        self.link_end_times = {}


    '''
    set_message_buffers() - set message buffers connected with HMCs
    @in_message_buffers: message buffers for incoming messages
    @out_message_buffers: message buffers for outgoing messages
    '''
    def set_message_buffers(self, in_message_buffers, out_message_buffers):
        self.in_message_buffers = in_message_buffers
        self.out_message_buffers = out_message_buffers
    # end of set_message_buffers

    def set_parameters(self, reduce_scatter_time_track_dict, all_gather_time_track_dict, per_message_max_latency, links_usage, link_start_times, link_end_times):
        self.reduce_scatter_time_track_dict = reduce_scatter_time_track_dict
        self.all_gather_time_track_dict = all_gather_time_track_dict
        self.per_message_max_latency = per_message_max_latency
        self.links_usage = links_usage
        self.link_start_times = link_start_times
        self.link_end_times = link_end_times


    '''
    schedule() - schedule the event at a given time
    @event: the event to be scheduled
    @cycle: scheduled time
    '''
    def schedule(self, event, cycle):
        self.global_eventq.schedule(self, cycle)
    # end of schedule()


    '''
    process() - event processing function in a particular cycle
    @cur_cycle: the current cycle that with events to be processed
    '''
    def process(self, cur_cycle):
        # send messages
        for i in range(self.args.num_hmcs):
            for j in range(self.args.radix):
                message = self.in_message_buffers[i][j].peek(cur_cycle)
                if message != None:
                    src = message.src // self.args.radix
                    src_ni = message.src % self.args.radix
                    dest = message.dest // self.args.radix
                    dest_ni = message.dest % self.args.radix
                    assert src == i
                    assert src_ni == j
                    msg_id = self.booksim.IssueMessage(message.flow, message.src, message.dest, message.id, message.size, message.type, message.submsgtype, message.priority, message.end)
                    if msg_id == -1:
                        self.schedule(self, cur_cycle + 1)
                        continue
                    self.in_message_buffers[i][j].dequeue(cur_cycle)
                    if (src, dest) not in self.link_start_times.keys():
                        self.link_start_times[src, dest] = []
                    self.link_start_times[src, dest].append(cur_cycle)
                    #print('{} | {} | issues a {} message for flow {} from HMC-{} (NI {}) to HMC-{} (NI {})'.format(cur_cycle, self.name, message.type, message.flow, src, src_ni, dest, dest_ni))

        self.booksim.SetSimTime(cur_cycle)
        self.booksim.WakeUp()

        # peek and receive messages
        for i in range(self.args.num_hmcs):
            for j in range(self.args.radix):
                dest_node = i * self.args.radix + j
                flow, src_node, msgtype, end, priority = self.booksim.PeekMessage(dest_node, 0)
                if src_node != -1:
                    assert flow != -1
                    src = src_node // self.args.radix
                    src_ni = src_node % self.args.radix
                    if self.out_message_buffers[i][j].is_full():
                        continue
                    message = Message(flow, None, src_node, dest_node, self.args.sub_message_size, msgtype, None, priority, end)
                    self.out_message_buffers[i][j].enqueue(message, cur_cycle, 1)
                    self.booksim.DequeueMessage(dest_node, 0)
                    if (src, i) not in self.link_end_times.keys():
                        self.link_end_times[src, i] = []
                    self.link_end_times[src, i].append(cur_cycle)
                    #print('{} | {} | peek a {} message for flow {} to HMC-{} (NI {}) from HMC-{} (NI {})'.format(cur_cycle, self.name, msgtype, flow, i, j, src, src_ni))

        if not self.booksim.Idle():
            self.schedule(self, cur_cycle + 1)
    # end of process()

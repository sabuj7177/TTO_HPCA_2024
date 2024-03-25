import math

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from easypyplot import barchart, pdf
from easypyplot import format as fmt

# plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.size'] = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def add_line(ax, xpos, ypos):
    line = plt.Line2D(
        #[xpos, xpos], [ypos + linelen, ypos],
        [xpos, xpos],
        [0, ypos],
        transform=ax.transAxes,
        color='black',
        linewidth=1)
    line.set_clip_on(False)
    ax.add_line(line)
    # ax.legend("AR Speedup")

def get_list(nn):
    if nn == "AlphaGoZero":
        layer_list = ["0", "1", "2_3", "4_5_6", "7"]
    elif nn == "FasterRCNN":
        layer_list = [
            "0_1_2", "3_4_5_6", "7_8_9", "10_11", "12", "13", "14", "15", "16", "17",
            "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43"
        ]
    elif nn == "NCF_recommendation":
        layer_list = ["0", "1", "2", "3_4_5_6_7"]
    elif nn == "Resnet152":
        layer_list = [
            "0_1_2_3_4", "5_6_7_8_9", "10_11_12_13_14", "15_16_17_18_19", "20_21_22_23_24",
            "25_26_27_28_29", "30_31_32_33_34", "35_36_37_38_39", "40_41_42_43_44", "45_46_47_48_49",
            "50_51_52_53_54", "55_56_57_58_59", "60_61_62_63_64", "65_66_67_68_69", "70_71_72_73_74",
            "75_76_77_78_79", "80_81_82_83_84", "85_86_87_88_89", "90_91_92_93_94", "95_96_97_98_99",
            "100_101_102_103_104", "105_106_107_108_109", "110_111_112_113_114", "115_116_117_118_119",
            "120_121_122_123_124", "125_126_127_128_129", "130_131_132_133_134", "135_136_137_138_139",
            "140_141_142_143_144", "145_146_147_148_149", "150_151_152_153_154", "155"
        ]
    elif nn == "Transformer":
        layer_list = [
            "0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19",
            "20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39",
            "40_41_42_43_44_45_46_47_48_49_50_51_52_53_54_55_56_57_58_59",
            "60_61_62_63_64_65_66_67_68_69_70_71_72_73_74_75_76_77_78_79",
            "80_81_82_83_84_85_86_87_88_89_90_91_92_93_94_95_96_97_98_99",
            "100_101_102_103_104_105_106_107_108_109_110_111_112_113_114_115_116_117_118_119",
            "120_121_122_123_124_125_126_127_128_129_130_131_132_133_134_135_136_137_138_139",
            "140_141_142_143_144_145_146_147_148_149_150_151_152_153_154_155_156_157_158_159",
            "160_161_162_163_164_165_166_167_168_169_170_171_172_173_174_175_176_177_178_179",
            "180_181_182_183_184_185_186_187_188_189_190_191_192_193_194_195_196_197_198_199",
            "200_201_202_203_204_205_206_207_208_209_210_211_212_213_214_215_216_217_218_219",
            "220_221_222_223_224_225_226_227_228_229_230_231_232_233_234_235_236_237_238_239",
            "240_241_242_243_244_245_246_247_248_249_250_251_252_253_254_255_256_257_258_259",
            "260_261_262_263_264_265_266_267_268_269_270_271_272_273_274_275_276_277_278_279",
            "280_281_282_283_284_285_286_287_288_289_290_291_292_293_294_295_296_297_298_299",
            "300_301_302_303_304_305_306_307_308_309_310_311_312_313_314_315_316_317_318_319",
            "320_321_322_323_324_325_326_327_328_329_330_331_332_333_334_335_336_337_338_339",
            "340_341_342_343_344_345_346_347_348_349_350_351_352_353_354_355_356_357_358_359",
            "360_361_362_363_364_365_366_367_368_369_370_371_372_373_374_375_376_377_378_379",
            "380_381_382_383_384_385_386_387_388_389_390_391_392_393_394_395_396_397_398_399",
            "400_401_402_403_404_405_406_407_408_409_410_411_412_413_414_415_416_417_418_419",
            "420_421_422_423_424_425_426_427_428_429_430_431_432_433_434_435_436_437_438_439",
            "440_441_442_443_444_445_446_447_448_449_450_451_452_453_454_455_456_457_458_459",
            "460_461_462_463_464_465_466_467_468_469_470_471_472_473_474_475_476_477_478_479",
            "480_481_482_483_484_485_486_487_488_489_490_491_492_493_494_495_496_497_498_499",
            "500_501_502_503_504_505_506_507_508_509_510_511_512_513_514_515_516_517_518_519",
            "520_521_522_523_524_525_526_527_528_529_530_531_532_533_534_535_536_537_538_539",
            "540_541_542_543_544_545_546_547_548_549_550_551_552_553_554_555_556_557_558_559",
            "560_561_562_563_564_565_566_567_568_569_570_571_572_573_574_575_576_577_578_579",
            "580_581_582_583_584_585_586_587_588_589_590_591_592_593_594_595_596_597_598_599",
            "600_601_602_603_604_605_606_607_608_609_610_611_612_613_614_615_616_617_618_619",
            "620_621_622_623_624_625_626_627_628_629_630_631_632_633_634_635_636_637_638_639",
            "640_641_642_643_644_645_646_647_648_649_650_651_652_653_654_655_656_657_658_659",
            "660_661_662_663_664_665_666_667_668_669_670_671_672_673_674_675_676_677_678_679",
            "680_681_682_683_684_685_686_687_688_689_690_691_692_693_694_695_696_697_698_699",
            "700_701_702_703_704_705_706_707_708_709_710_711_712_713_714_715_716_717_718_719",
            "720_721_722_723_724_725_726_727_728_729_730_731_732_733_734_735_736_737_738_739",
            "740_741_742_743_744_745_746_747_748_749_750_751_752_753_754_755_756_757_758_759",
            "760_761_762_763_764_765_766_767_768_769_770_771_772_773_774_775_776_777_778_779",
            "780_781_782_783_784_785_786_787_788_789_790_791_792_793_794_795_796_797_798_799",
            "800_801_802_803_804_805_806_807_808_809_810_811_812_813_814_815_816_817_818_819",
            "820_821_822_823_824_825_826_827_828_829_830_831_832_833_834_835_836_837_838_839",
            "840_841_842_843_844_845_846_847_848_849_850_851_852_853_854_855_856_857_858_859",
            "860_861_862_863_864_865_866_867_868_869_870_871_872_873_874_875_876_877_878_879",
            "880_881_882_883_884_885_886_887_888_889_890"
        ]
    elif nn == "alexnet":
        layer_list = ["0", "1", "2", "3", "4"]
    else:
        layer_list = [
            "0_1_2", "3_4_5", "6_7_8_9", "10", "11", "12_13", "14_15", "16", "17",
            "18_19_20_21", "22", "23", "24_25", "26", "27", "28", "29", "30_31_32",
            "33", "34", "35", "36_37", "38", "39", "40", "41", "42_43", "44", "45",
            "46", "47", "48_49", "50", "51", "52", "53", "54", "55", "56", "57"]
    return layer_list


def plot_overlap(folder_path, names, total_nodes, scalesim_config, train_folder_path, ax, schemes):
    benchmarks = ['alexnet', 'AlphaGoZero', 'FasterRCNN', 'Googlenet', 'NCF_recommendation', 'Resnet152', 'Transformer']
    # benchmarks = ['Googlenet']

    entry_names = ['Communication', 'Computation-Communication Overlap', 'Computation']
    xlabels = ['AlexNet', 'AlphaGoZero', 'FasterRCNN', 'GoogLeNet', 'NCF', 'ResNet152', 'Transformer']
    group_names = []

    cycles = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=float)
    norm_cycles = np.zeros(
        (int(len(schemes)), int(len(xlabels))), dtype=float)
    norm_allreduce_cycles = np.zeros(
        (int(len(schemes)), int(len(xlabels))), dtype=float)
    just_training_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=float)
    just_allreduce_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=float)
    overlap_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=float)
    cycles_breakdown = np.zeros((3, int(len(benchmarks) * len(schemes))), dtype=float)
    norm_cycles_breakdown = np.zeros((3, int(len(benchmarks) * len(schemes))), dtype=float)
    total_imagenet_data = 1281167

    for s, name in enumerate(names):
        for b, bench in enumerate(benchmarks):
            filename = "%s/%s_%s_%d_mesh_%s_%s.json" % (train_folder_path, bench, 'mesh_overlap_2d_1', total_nodes, bench, scalesim_config)

            if os.path.exists(filename):
                with open(filename, 'r') as json_file:
                    sim = json.load(json_file)
                    training_cycles = sim['results']['performance']['training_by_layer']
                    total_training_time = sim['results']['performance']['total']
                    # print(training_cycles)
                    json_file.close()
            else:
                print("Missing")


            layer_list = get_list(bench)
            training_cycles_list = []
            for layer in layer_list[::-1]:
                first_layer_number = layer.split('_')[0]
                training_cycles_list.append(int(training_cycles[first_layer_number]))

            remainder = total_training_time - training_cycles_list[-1]
            allreduce_cycles_list = []
            for layer in layer_list[::-1]:
                if name == 'multitree' or name == 'ring_bi' or name == 'ring' or name == 'ring_odd' or name == 'ring_odd_bi' or name == 'dtree' or name == 'ring2dn':
                    filename = "%s/%s/%s_%s_%d_mesh_200_%s_%s_layer_%s.json" % (folder_path, bench, bench, name, total_nodes, bench, scalesim_config, layer)
                else:
                    filename = "%s/%s/%s_%s_%d_mesh_%s_%s_layer_%s.json" % (folder_path, bench, bench, name, total_nodes, bench, scalesim_config, layer)

                if os.path.exists(filename):
                    with open(filename, 'r') as json_file:
                        sim = json.load(json_file)
                        allreduce_cycles_list.append(int(sim['results']['performance']['allreduce']['total']))
                        # print(training_cycles)
                        json_file.close()
                else:
                    print("Missing " + name + " " + bench + " " + layer)
            assert len(training_cycles_list) == len(allreduce_cycles_list)
            total_time = training_cycles_list[0]
            training_cycles_list.pop(0)
            total_time += allreduce_cycles_list[0]
            total_allreduce_time = allreduce_cycles_list[0]
            allreduce_cycles_list.pop(0)
            remaining_layers = len(training_cycles_list)
            for i in range(remaining_layers):
                if training_cycles_list[0] > total_time:
                    total_time = training_cycles_list[0]
                training_cycles_list.pop(0)
                total_time += allreduce_cycles_list[0]
                total_allreduce_time += allreduce_cycles_list[0]
                allreduce_cycles_list.pop(0)
            assert len(training_cycles_list) == 0
            assert len(allreduce_cycles_list) == 0
            total_time += remainder

            # print("Final")
            just_training = total_time - total_allreduce_time
            overlap = total_training_time - just_training
            just_allreduce = total_allreduce_time - overlap

            if name == 'mesh_overlap_2d_1':
                total_iteration = math.ceil(total_imagenet_data / ((total_nodes - 1) * 16))
            else:
                total_iteration = math.ceil(total_imagenet_data / (total_nodes * 16))

            just_allreduce_cycles[s][b] = just_allreduce * total_iteration
            just_training_cycles[s][b] = just_training * total_iteration
            overlap_cycles[s][b] = overlap * total_iteration
            cycles[s][b] = total_time * total_iteration

            norm_cycles[s][b] = cycles[s][b] / cycles[0][b]
            cycles_breakdown[0][b * len(schemes) + s] = just_allreduce_cycles[s][b]
            cycles_breakdown[1][b * len(schemes) + s] = overlap_cycles[s][b]
            cycles_breakdown[2][b * len(schemes) + s] = just_training_cycles[s][b]

    speedup = 1.0 / norm_cycles
    speedup[np.isnan(speedup)] = 0

    for b, bench in enumerate(benchmarks):
        for s, name in enumerate(names):
            group_names.append(schemes[s])
            for e, entry in enumerate(entry_names):
                norm_cycles_breakdown[e][b * len(schemes) + s] = cycles_breakdown[e][b * len(schemes) + s] / cycles[0][
                    b]
    norm_cycles_breakdown[np.isnan(norm_cycles_breakdown)] = 0

    colors = ['#B2A4FF', '#A4D0A4', '#BE5A83', '#F94A29']
    xticks = []
    for i in range(0, len(benchmarks)):
        for j in range(0, len(schemes)):
            xticks.append(i * (len(schemes) + 1) + j)
    data = [list(i) for i in zip(*norm_cycles_breakdown)]
    data = np.array(data, dtype=np.float64)
    hdls = barchart.draw(
        ax,
        data,
        group_names=group_names,
        entry_names=entry_names,
        breakdown=True,
        xticks=xticks,
        width=0.8,
        colors=colors,
        legendloc='upper center',
        legendncol=len(entry_names),
        xticklabelfontsize=20,
        xticklabelrotation=90,
        log=False)

    ax.set_ylabel('Normalized Training Time')
    ax.yaxis.grid(True, linestyle='--')
    fmt.resize_ax_box(ax, hratio=0.95)
    ly = len(benchmarks)
    scale = 1. / ly
    ypos = -.5
    for pos in range(ly + 1):
        lxpos = (pos + 0.5) * scale
        if pos < ly:
            ax.text(
                lxpos, ypos, xlabels[pos], ha='center', transform=ax.transAxes)
        add_line(ax, pos * scale, ypos)
    temp_legend = ax.get_legend()
    ax.get_legend().remove()
    ax.tick_params(axis='both')
    ax.set_ylim(0, 2.5)
    return temp_legend


def main():
    plt.rcParams["figure.figsize"] = [14.00, 7.0]
    plt.rcParams["figure.autolayout"] = True
    figure, ax1 = plt.subplots(1, 1)

    folder_path = '{}/HPCA_2024_final/overlap/comm'.format(os.environ['SIMHOME'])
    train_folder_path = '{}/HPCA_2024_final/overlap/train'.format(os.environ['SIMHOME'])
    even_names = ['ring', 'ring2dn', 'dtree', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']
    schemes_evens = ['Ring', 'Ring-2D', 'DTree', 'Multitree', 'RingBiEven', 'TTO']
    legend = plot_overlap(folder_path, even_names, 64, 'google', train_folder_path, ax1, schemes_evens)

    lines_labels = [ax1.get_legend_handles_labels()]
    labels = [] if legend is None else [str(x._text) for x in legend.texts]
    handles = [] if legend is None else legend.legendHandles
    figure.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
    figure.savefig('overlap.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()

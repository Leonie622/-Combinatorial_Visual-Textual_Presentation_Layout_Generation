from lib2to3.pgen2.pgen import generate_grammar
import os
from PIL import Image

from lib import PosterGenerationEngine

if __name__ == '__main__':
    os.system('mkdir -p output')

    engine = PosterGenerationEngine()

    # Example 1~6 不设置额外参数，可以完美生成模板的原始效果图

    engine.render(
        engine.match_template('fashion'),
        title='柠檬苏打水',
        subtitle='无糖0卡0负担。\n矿泉水新花招，\n气泡充足浓郁，\n享受轻松闲暇片刻。\n',
    ).save('output/example1.png')

    engine.render(
        engine.match_template('technology'),
        title='柠檬苏打水',
        subtitle='无糖0卡0负担。矿泉水新花招，气泡充足浓郁，享受轻松闲暇片刻。', # 可以不加换行符，程序会根据模板自动分割
    ).save('output/example2.png')

    engine.render(
        engine.match_template('cute'),
        title='柠檬苏打水',
        subtitle='无糖0卡0负担。矿泉水新花招，气泡充足浓郁，享受轻松闲暇片刻。',
    ).save('output/example3.png')

    engine.render(
        engine.match_template('vintage'),
        title='柠檬苏打水',
        subtitle='感受生命活力\n随时随地补给',
    ).save('output/example4.png')

    engine.render(
        engine.match_template('nature'),
        title='柠檬苏打水',
        subtitle='无糖0卡0负担\n矿泉水新花招\n气泡充足浓\n享受轻松闲暇片刻',
    ).save('output/example5.png')

    engine.render(
        engine.match_template('clean'),
        title='柠檬苏打水',
        subtitle='无糖0卡0负担，\n矿泉水新花招，\n气泡充足浓郁，\n享受轻松闲暇片刻。',
    ).save('output/example6.png')

    
    # Example 7 - 换色 / 换字 / 换图片

    engine.render(
        engine.match_template('fashion'),
        title='无糖可乐', # 可以替换 title, 字数可以不同
        subtitle='快乐饮料，\n健康之选，\n无糖0卡0负担。\n气泡充足浓郁，\n享受轻松闲暇片刻。\n', # 可以替换 subtitle, 字数和行数都可以不同
        image=Image.open('static/cola.png'), # 对应 image layer，支持模板中的 mask，程序会自适应调整尺寸
        colors={
            'bg': '#FFEFD5', # 对应 background layer
            'bg2': '#F08080', # 对应 background layer2
            'ex': '#8B658B', # 对应 embellishment layer
            'title': '#6A5ACD', # 对应 title layer
            'subtitle': '#3c3c3c', # 对应 subtitle layer
        },
    ).save('output/example7.png')

    # Example 8 - 换竖排文字

    engine.render(
        engine.match_template('clean'),
        title='无糖可乐',
        subtitle='快乐饮料，\n健康之选，\n无糖0卡0负担。\n气泡充足浓郁，\n享受轻松闲暇片刻。\n', # 替换竖排文字
        image=Image.open('static/cola.png'),
    ).save('output/example8.png')

    # Example 9 - 替换后文字居中 or 填满整行属性保持不变

    engine.render(
        engine.match_template('nature'),
        title='无糖冰红茶',
        subtitle='快乐饮料，\n健康之选，\n无糖0卡0负担。\n气泡充足浓郁，\n享受轻松闲暇片刻。\n',
        image=Image.open('static/cola.png'),
        colors={
            'title': 'white'
        },
    ).save('output/example9.png')

    # Example 10 - 替换字体（可用的字体名称为对应模板下的 ttf 文件）；根据语料生成商品名

    engine.render(
        engine.match_template('cute'),
        title=engine.generate_title('红茶'),
        subtitle='感受生命活力\n随时随地补给\n',
        image=Image.open('static/cola.png'),
        fonts={
            'title': '汉仪悠然体简',
            'subtitle': '汉仪悠然体简',
        },
    ).save('output/example10.png')

from segment.segment import Segment

segment = Segment()

content = '百年来，我们党始终筑牢“一切为了人民”这一基石。“一切为了人民”是中国共产党保持旺盛生机的密码。十九届五中全会建议稿中突出强调了“扎实推动共同富裕”，这样表述，这在党的全会文件中还是第一次，彰显了“发展为人民”的理念。“江山就是人民，人民就是江山”，中国共产党每一个历史转折，每一个伟大胜利背后，都蕴藏着巨大的人民力量。一百年来，党始终与广大人民群众同舟共济、生死与共，在革命、建设、改革的风雨考验中，矢志不渝为了人民，中国“红船”才能勇往直前，击鼓催征稳驭舟。'
print('content: ', content)

print('seg(content)')
words = list(segment.seg(content))
print(words)

print('seg(content, model=\'HMM\')')
words = list(segment.seg(content, model='HMM'))
print(words)

print('seg(content, model=\'CRF\')')
words = list(segment.seg(content, model='CRF'))
print(words)

print('seg(content, model=\'DL\')')
words = list(segment.seg(content, model='DL'))
print(words)

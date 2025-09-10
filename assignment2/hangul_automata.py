# -*- coding: utf-8 -*- 

CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
       'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ',
        'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
        'ㅡ', 'ㅢ', 'ㅣ']
JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
        'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
        'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def is_cho(char): return char in CHO
def is_jung(char): return char in JUNG
def is_jong_candidate(char): return char in CHO

class HangulAutomata:
    def __init__(self):
        self.reset()
        self.result = []

    def reset(self):
        self.state = 0
        self.ch = ''
        self.ju = ''
        self.jo = ''

    def flush(self):
        if self.state == 1:
            self.result.append(self.ch)
        elif self.state in [2, 3]:
            self.result.append(self.compose())
        self.reset()

    def compose(self):
        try:
            cho = CHO.index(self.ch)
            jung = JUNG.index(self.ju)
            jong = JONG.index(self.jo) if self.jo in JONG else 0
            return chr(0xAC00 + cho * 21 * 28 + jung * 28 + jong)
        except:
            return self.ch + self.ju + self.jo

    def backspace(self):
        if self.state == 3 and self.jo:
            self.jo = ''
            self.state = 2
        elif self.state == 2 and self.ju:
            self.ju = ''
            self.state = 1
        elif self.state == 1 and self.ch:
            self.ch = ''
            self.state = 0
        elif self.state == 0 and self.result:
            self.result.pop()

    def process(self, text):
        i = 0
        while i < len(text):
            c = text[i]
            next_c = text[i + 1] if i + 1 < len(text) else ''

            if c == '<':
                self.backspace()
                i += 1
                continue

            if c.isdigit():
                self.flush()
                self.result.append(c)
                i += 1
                continue

            if self.state == 0:
                if is_cho(c):
                    self.ch = c
                    self.state = 1
                else:
                    self.result.append(c)

            elif self.state == 1:
                if is_jung(c):
                    self.ju = c
                    self.state = 2
                elif is_cho(c):
                    self.result.append(self.ch)
                    self.ch = c
                    self.state = 1
                else:
                    self.result.append(self.ch)
                    self.result.append(c)
                    self.reset()

            elif self.state == 2:
                if is_cho(c):
                    combined = self.jo + c
                    if next_c and is_jung(next_c):
                        # 다음 입력이 모음이면 초성으로 간주
                        self.flush()
                        self.ch = c
                        self.state = 1
                    else:
                        # 복합종성 여부 확인
                        if (self.jo + c) in JONG:
                            self.jo = self.jo + c
                            self.state = 3
                        else:
                            self.jo = c
                            self.state = 3
                else:
                    self.flush()
                    self.result.append(c)

            elif self.state == 3:
                if is_jung(c):
                    # 종성을 초성으로 이동
                    self.flush()
                    self.ch = self.jo[-1]  # 두 번째 자음으로 새로 시작
                    self.ju = c
                    self.jo = ''
                    self.state = 2
                else:
                    self.flush()
                    self.ch = c
                    self.state = 1
            i += 1

    def get_text(self):
        self.flush()
        return ''.join(self.result)

if __name__ == '__main__':
    automata = HangulAutomata()
    try:
        while True:
            text = input("자소를 입력하세요. (Ctrl+C로 종료)\n")
            automata = HangulAutomata()
            automata.process(text)
            print(automata.get_text())
    except KeyboardInterrupt:
        print("\n종료합니다.")

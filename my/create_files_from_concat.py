import re

story_reg = re.compile(r"(?<=<\|startoftext\|>).*?(?=<\|endoftext\|>)", re.DOTALL)


def main():
    with open(r"X:\source\PycharmProjects\gpt-2-tf2-p6\content\dungeon_text.txt", "r", encoding="utf-8") as fp:
        data = fp.read()
    stories = story_reg.findall(data)
    print(stories)
    for i, story in enumerate(stories):
        with open(fr"X:\source\PycharmProjects\gpt-2-tf2-p6\gpt2-tf2\data\ai_dungeon_data\\{i}.txt", "w", encoding="utf-8") as fp:
            fp.write(story)


if __name__ == '__main__':
    main()




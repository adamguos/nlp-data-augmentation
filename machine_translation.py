# Goal: Alter reddit data with
from DataLoader import DataLoader
from googletrans import Translator
from icecream import ic


if __name__ == "__main__":
    dl = DataLoader(verbose=True)
    t = Translator()

    x, y = dl.import_unaltered_reddit(5000)

    languages = ["af", "zh-cn", "fr", "hr", "ja", "la", "no", "es", "sw", "zu"]
    english = "en"

    with open("mt_reddit.txt", "w", encoding="utf-8") as f:

        for i in range(len(x)):

            for l in languages:
                translated = t.translate(text=x[i], dest=l, src=english).text
                back_in_eng = t.translate(text=translated, dest=english, src=l).text

                f.write(f"{y[i]}\t{back_in_eng}\n")

Bu proje ne işe yarıyor?

Seçilen şairin tarzını öğrenip onun şiirlerine benzer bir başka yeni şiir üretecek. 
-Bir metni okuyup, benzer tarzda yeni metin üreten bir yapay sinir ağı geliştirmek.

Model, verilen metni yeniden yazmaz veya dönüştürmez.
Bunun yerine, verilen metni bir seed (başlangıç) olarak alır ve seçilen şairden öğrendiği stilistik örüntüleri kullanarak metni devam ettirir


Aşağıdaki stilistik özellikleri öğrenir:
kelime dağılımı
cümle uzunluğu
noktalama işaretleri
ritim ve yapısal örüntüler
Öğrenilen tarzda yeni şiirsel metinler üretir.


Desteklenen şairler 
William Shakespeare
Emily Dickinson
Walt Whitman

Neden RNN / LSTM?
Metin = sıralı veri
Önceki kelimeler → sonraki kelimeyi etkiler
RNN bu bağımlılığı modellemek için var


Klasik RNN → uzun seq → gradyan küçülür → geçmişi öğrenemez
LSTM       → gate + cell → gradyan korunur → geçmişi hatırlar

Çalışma Mantığı
1-character → embedding → vektör
2-LSTM → hidden + cell state ile geçmişi hatırlar
3-Linear layer → tüm karakterler için olasılık çıkarır
4-Softmax → en olası karakteri tahmin eder
5-Bu tahmin → yeni input olarak kullanılabilir (text generation)


Whitman şiirleri, character-level bir model için oldukça uzun olduğu için ek bir 
ön işleme adımı gerekti. Bu noktada, tüm şiirleri tek satır halinde aldıktan sonra 
split_whitman.py adında küçük bir Python scripti yazarak metni modelin 
daha rahat öğrenebileceği parçalara böldüm


proje run edilirken:
1- train.py'ı tek tek terminalden çalıştır:

python train.py --data data/dickinson_poems.txt --out models/dickinson_model.pt
python train.py --data data/whitman_poems.txt --out models/whitman_model.pt
python train.py --data data/shakespeare_poems.txt --out models/shakespeare_model.pt

2- terminalde streamlit run edilmeli:
 streamlit run app.py
streamlit generate.py dosyasını çalıştıracak.


load txt özelliğini test etmek için benim şiirimin full hali:
istenilen yerden parçalanabilir.

Love
Is that you?
There’s nothing much more than pain and happiness.
The feeling that came along with him, and the flashes,
La vie en rose playing in the background,
In the middle of a sunset.
He sticks in your mind…
When you feel lost, your balance — and —
That moment you get it: you fell in love with
Who you shouldn’t be.
But when he smiles at you,
You understand what the real devil is.
It seems like an angel when it smiles at you.
There’s one thing that I wanted
From that devil:
Make me happy and blue
Forever.

medium link: 
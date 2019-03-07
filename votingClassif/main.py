import sent_mod as sent
# import pickle
import statistics

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

pos_paragraphs = short_pos.split('\n')
neg_paragraphs = short_neg.split('\n')

test_pos = pos_paragraphs[5000:]
test_neg = neg_paragraphs[5000:]
test_set_len = len(test_neg) + len(test_pos)

print("Running Voting Classifiers...")

error_count = 0
error_count2 = 0
try:
    for p in pos_paragraphs:
        if sent.sentiment(p)[0] != 'pos':
            error_count += 1
    for p in neg_paragraphs:
        if sent.sentiment(p)[0] != 'neg':
            error_count += 1
except statistics.StatisticsError:
    error_count2 +=1
    pass

error_count2 += error_count
accuracy1 = ((1 - error_count / test_set_len) * 100)
accuracy2 = ((1 - error_count2 / test_set_len) * 100)
print("Voting CLassifiers accuracy ranging form: ", accuracy2, ' to ', accuracy1)

# To test it yourself
# print(type(sent.sentiment("This movie was awesome!")))
# # print(sent.sentiment("This movie was aweful"))

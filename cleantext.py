#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import json
import re
import string
import argparse


__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
	"tis": "'tis",
	"aint": "ain't",
	"amnt": "amn't",
	"arent": "aren't",
	"cant": "can't",
	"couldve": "could've",
	"couldnt": "couldn't",
	"didnt": "didn't",
	"doesnt": "doesn't",
	"dont": "don't",
	"hadnt": "hadn't",
	"hasnt": "hasn't",
	"havent": "haven't",
	"hed": "he'd",
	"hell": "he'll",
	"hes": "he's",
	"howd": "how'd",
	"howll": "how'll",
	"hows": "how's",
	"id": "i'd",
	"ill": "i'll",
	"im": "i'm",
	"ive": "i've",
	"isnt": "isn't",
	"itd": "it'd",
	"itll": "it'll",
	"its": "it's",
	"mightnt": "mightn't",
	"mightve": "might've",
	"mustnt": "mustn't",
	"mustve": "must've",
	"neednt": "needn't",
	"oclock": "o'clock",
	"ol": "'ol",
	"oughtnt": "oughtn't",
	"shant": "shan't",
	"shed": "she'd",
	"shell": "she'll",
	"shes": "she's",
	"shouldve": "should've",
	"shouldnt": "shouldn't",
	"somebodys": "somebody's",
	"someones": "someone's",
	"somethings": "something's",
	"thatll": "that'll",
	"thats": "that's",
	"thatd": "that'd",
	"thered": "there'd",
	"therere": "there're",
	"theres": "there's",
	"theyd": "they'd",
	"theyll": "they'll",
	"theyre": "they're",
	"theyve": "they've",
	"wasnt": "wasn't",
	"wed": "we'd",
	"wedve": "wed've",
	"well": "we'll",
	"were": "we're",
	"weve": "we've",
	"werent": "weren't",
	"whatd": "what'd",
	"whatll": "what'll",
	"whatre": "what're",
	"whats": "what's",
	"whatve": "what've",
	"whens": "when's",
	"whered": "where'd",
	"wheres": "where's",
	"whereve": "where've",
	"whod": "who'd",
	"whodve": "whod've",
	"wholl": "who'll",
	"whore": "who're",
	"whos": "who's",
	"whove": "who've",
	"whyd": "why'd",
	"whyre": "why're",
	"whys": "why's",
	"wont": "won't",
	"wouldve": "would've",
	"wouldnt": "wouldn't",
	"yall": "y'all",
	"youd": "you'd",
	"youll": "you'll",
	"youre": "you're",
	"youve": "you've"
}

def sanitize(text):
	"""Do parse the text in variable "text" according to the spec, and return
	a LIST containing FOUR strings
	1. The parsed text.
	2. The unigrams
	3. The bigrams
	4. The trigrams
	"""
	# Replace newline and tabs
	temp_parsed = re.sub("[\n\t]", " ", text)

	# Replace URLs with the text inside the square brackets
	temp_parsed = re.sub(r"\[([\w\s\.]+)\]\(https?://[\w\s\.\-\#=/]+\)",
							r"\g<1>", temp_parsed)
	
	# Replace any URLs with single space
	temp_parsed = re.sub(r"https?://[\w\s\.\-/]+",
							r" ", temp_parsed)

	# Split text on single space
	# 	If there are multiple contiguous spaces
	# 	clean the resulting list of empty string tokens
	temp_parsed = [token for token in temp_parsed.split(" ") if token != ""]

	# Separate external punctuation:
	# 	converts token to lowercase.
	# 	leaves simple tokens (no external punctuation) as-is
	# 	splits complex (non-simple) tokens into 3-lists as follows:
	# 	list(t[0]) = split beginning punc, t[1] = string, list(t[2]) = split endpunc
	# 	e.g. "!!!AB##$cD!" --> list(t[0]) = ['!', '!', '!'], t[1] = "ab##$cd", list(t[2]) = ['!']
	# 	note: [#$%'] are considered valid non-punctuation characters from website
	# 	note: for "/r/subreddit" will return "r/subreddit" (acceptable, according to spec)
	split_parsed = []
	for token in temp_parsed:
		t = re.split("([a-z0-9#$%'].*[a-z0-9#$%'])", str.lower(token))
		# print(t)
		if len(t) == 3:
			splittoken = list(t[0]) + [t[1]] + list(t[2])
			split_parsed.extend(splittoken)
		else:
			split_parsed.extend(t)
	# print(split_parsed)

	# Removes all punctuation EXCEPT embedded AND terminating punctuation
	# i.e. remove all standalone nonalnum tokens that are not , . ! ? ; :
	# After this, list of parsed tokens will be complete.
	common_punc = set([".", "!", "?", ",", ";", ":"])
	def invalid_token(token):
		all_invalid = 1
		for char in list(token):
			if char.isalnum() or char in common_punc:
				all_invalid = 0
		return all_invalid
		
	parsed = [token for token in split_parsed if not invalid_token(token)]
	# print(parsed)

	# Build lists of n-grams
	unigrams = [token for token in parsed if token not in common_punc]

	bigrams = []
	for i in range(len(parsed)):
		if (i + 1) < len(parsed) \
		and parsed[i] not in common_punc \
		and parsed[i+1] not in common_punc:
			bigrams.append(parsed[i] + "_" + parsed[i+1])

	trigrams = []
	for i in range(len(parsed)):
		if (i + 2) < len(parsed) \
		and parsed[i] not in common_punc \
		and parsed[i+1] not in common_punc \
		and parsed[i+2] not in common_punc:
			trigrams.append(parsed[i] + "_" + parsed[i+1] + "_" + parsed[i+2])

	# Join lists into strings
	parsed_str = " ".join(parsed)
	uni_str = " ".join(unigrams)
	bi_str = " ".join(bigrams)
	tri_str = " ".join(trigrams)
	return [parsed_str, uni_str, bi_str, tri_str]

# Usage: python3 cleantext.py sample.json
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("inp_filename")
	args = parser.parse_args()
	with open(args.inp_filename, "r") as json_file:
		counter = 1
		for json_comment in json_file:
			# COUNTER = LINE NUMBER IN sample.json
			if counter == 4853: 
				data = json.loads(json_comment)
				results = sanitize(data["body"])
				# print("----INPUT----")
				# print(data["body"])
				# print("----PARSED----")
				# print(results[0])
				# print("----UNIGRAM----")
				# print(results[1])
				# print("----BIGRAM----")
				# print(results[2])
				# print("----TRIGRAM----")
				# print(results[3])
				break
			else:
				counter += 1
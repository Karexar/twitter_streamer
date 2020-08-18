import re
from typechecker.typecheck import *
import unidecode

class Cleaner:
    space_regexs = [(r"\s+", " "), (r"\s,", ","), (r"\s\.", ".")]

    good_chars = " !\"$%&\'()*+,-./0123456789:;?"
    good_chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    good_chars += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
    good_chars = set(good_chars)

    html_entities = ["&lt;", "&gt;", "&le;", "&ge;", "&amp;",
                     "&lt", "&gt", "&le", "&ge", "&amp"]

    smileys = ['-\\_(ツ)_/-', '\\_(ツ)_/', "(ツ)", '\\^o^/', '^o^', '^~^', "^^'",
               '^^', "^ ^'", '^ ^', '*-*', '*^*', '*~*', '*.*', "-.-'", '-.-',
               '\\m/', ':*', ':\\', ':/', ':0', ':)', ':(', ':D', ':p', ':P',
               ':d', ':o', ':O', ':-*', ':-\\', ':-/', ':-0', ':-)', ':-(',
               ':-D', ':-p', ':-P', ':-d', ':-o', ':-O', ';*', ';\\', ';/',
               ';0', ';)', ';(', ';D', ';p', ';P', ';d', ';o', ';O', ';-*',
               ';-\\', ';-/', ';-0', ';-)', ';-(', ';-D', ';-p', ';-P', ';-d',
               ';-o', ';-O', '=*', '=\\', '=/', '=0', '=)', '=(', '=D', '=p',
               '=P', '=d', '=o', '=O', '=-*', '=-\\', '=-/', '=-0', '=-)',
               '=-(', '=-D', '=-p', '=-P', '=-d', '=-o', '=-O', '=*', '=\\',
               '=/', '=0', '=)', '=(', '=D', '=p', '=P', '=d', '=o', '=O',
               '=-*', '=-\\', '=-/', '=-0', '=-)', '=-(', '=-D', '=-p', '=-P',
               '=-d', '=-o', '=-O', '<3', 'XD']

    @staticmethod
    @accepts(str, List[Tuple[str, str]])
    @returns(str)
    def preprocess(sentence, regexs):
        """Preprocess a given sentence by replacing specific patterns. This is
        useful e.g. for removing mentions, hashtags, urls, and so on.

        Parameters
            sentence | str
                The sentence to process
            regexs | List[Tuple[str, str]]
                A list of tuple with first element being the pattern to match
                and the second element being the string that replaces the
                pattern.
        """
        for regex, replacement in regexs:
            sentence = re.sub(regex, replacement, sentence)
        return sentence

    @staticmethod
    @accepts(str)
    @returns(str)
    def remove_smileys(sentence):
        """Remove smileys from text. This is different from the emojis because
        the smileys are made by combining characters (mostly ascii), e.g. :-)
        """
        for smiley in Cleaner.smileys:
            sentence = sentence.replace(smiley, "")
        return sentence

    @staticmethod
    @accepts(str)
    @returns(str)
    def remove_hat_element(sentence):
        """Remove some hat elements (^gf, ^chs, ^ds and so on) that appears
        sometimes at the end of sentences. The meaning and origin of these
        elements are unknown.
        """
        sentence = re.sub("\^\w+$", "", sentence)
        return sentence

    @staticmethod
    @accepts(str)
    @returns(str)
    def remove_html_entities(sentence):
        """Remove some html entities
        """
        for x in Cleaner.html_entities:
            sentence = sentence.replace(x, " ")
        return sentence

    @staticmethod
    @returns(str)
    def remove_wrong_chars(sentence):
        """Replace by a space all characters that are not in good_chars"""
        chars = set(list(sentence))
        for c in chars:
            if c not in Cleaner.good_chars:
                sentence = sentence.replace(c, " ")
        return sentence

    @staticmethod
    @returns(str)
    def remove_special_chars(sentence, special_chars=set("#@[]{}<>=^\\_~")):
        """Remove special characters from a list of sentences"""
        chars = set(list(sentence))
        for c in chars:
            if c in special_chars:
                sentence = sentence.replace(c, " ")
        return sentence

    @staticmethod
    @accepts(str)
    @returns(str)
    def remove_special_words(sentence):
        """Remove words containing one or more character that are not in
        the good character list"""
        final = []
        # add the <> to avoid removing the tags
        chars_ok = Cleaner.good_chars.union({'<', '>'})
        for word in sentence.split():
            if len(set(word).difference(chars_ok)) == 0:
                final.append(word)
        return ' '.join(final)

    @staticmethod
    @accepts(str)
    @returns(str)
    def clean_spaces(sentence):
        """Remove duplicated spaces, spaces before a dot or comma, and spaces
        at the beginning or end of a sentence"""

        sentence = Cleaner.preprocess(sentence, Cleaner.space_regexs)
        sentence = sentence.strip()
        return sentence

    def remove_duplicated_spaces(sentence):
        """Remove duplicated spaces from a given sentence"""
        sentence = re.sub("\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence

    def add_num_token(sentence):
        """Convert all numbers in a <num> token and isolate the
        token by adding a space before and after.
        """
        sentence = re.sub("[0-9][0-9\.\,\'/]*[0-9]", " <num> ", sentence)
        sentence = re.sub("\<num\>(\s+\<num\>)+", "<num>", sentence)
        return sentence

    def add_punc_token(sentence):
        """Convert all punctuation signs in a <punc> token and isolate the
        token by adding a space before and after. Make sure to run this after
        add_num_token.
        """
        return re.sub("[\.\,\:\;\?\!\]\(\)\"]+", " <punc> ", sentence)
        sentence = re.sub("\<num\>(\s+\<punc\>)+", "<punc>", sentence)
        return sentence

    def remove_special_duplication(sentence):
        """Remove the duplication of special characters
        """
        chars = set("$%&'*+-/")
        for c in chars:
            if c in sentence:
                sentence = re.sub("\\" + c + "+", c, sentence)
        return sentence

    def isolate_special_characters(sentence):
        """Isolate some special characters that do not form a word with its
        neighbors"""
        chars = set("$%&*+")
        for c in chars:
            if c in sentence:
                sentence = sentence.replace(c, " " + c + " ")
        return sentence

    def replace_non_gsw_accent(sentence):
        """Replace all non gsw accent by the letter without the accent"""
        res = []
        for c in sentence:
            if not c in {'Ö', 'Ü', 'ä', 'ö', 'ü'}:
                c = unidecode.unidecode(c)
            res.append(c)
        return ''.join(res)

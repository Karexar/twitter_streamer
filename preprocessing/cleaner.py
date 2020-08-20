import re
from typechecker.typecheck import *
import unidecode

class Cleaner:
    good_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    good_chars += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
    good_chars += " -,.?!0123456789%&\"\'()/$*+:;<=>[]\\^_{}|\\~€°²"
    good_chars = set(good_chars)

    html_entities = ["&lt;", "&gt;", "&le;", "&ge;", "&amp;",
                     "&lt", "&gt", "&le", "&ge", "&amp"]

    smileys = ['-\\_(ツ)_/-', '\\_(ツ)_/', "(ツ)", '\\^o^/', '^o^', '^~^', "^^'",
               '^^', "^ ^'", '^ ^', '*-*', '*^*', '*~*', '*.*', "-.-'", '-.-',
               '\\m/', '<3', ' XD', '^.^', '\\o/', '\\(. _. )/', '/o\\',
               '\\(\'o\')/', '\\(~_~)/', '\\*O*/', '\\/', '/\\',
               '-\\_( •-•)_/-', '-\\_( -)_/-', '-\\_(-)_/-', '-\\_(- )_/-']

    punc_set = set(".,:;!?")
    punc_set_no_period = set(",:;!?")

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
        regex = r"[\:\;\=]{1}-?([\\\/DpPdoO0\*)(\]\[\]])\1*"
        sentence = re.sub(regex, " ", sentence)
        regex = r"[\:\;\=]{1}\s?-\s?([\\\/\*)(\]\[\]])\1*"
        sentence = re.sub(regex, " ", sentence)

        for smiley in Cleaner.smileys:
            sentence = sentence.replace(smiley, " ")
        return sentence

    @staticmethod
    @accepts(str)
    @returns(str)
    def remove_hat_element(sentence):
        """Remove some hat elements (^gf, ^chs, ^ds and so on) that appears
        sometimes at the end of sentences. The meaning and origin of these
        elements are unknown.
        """
        sentence = re.sub(r"\^\w+$", "", sentence)
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
    @accepts(str, int, special=set)
    @returns(str)
    def remove_groups_of_special_chars(
                                sentence,
                                from_size,
                                special=set("-,%&\"'()/$*+:;<=>[]^_{}|\\~€°²")):
        """Remove tokens of consecutive special characters, if the length is
        at least 'from_size'. The tokens are space-separated
        """
        final_words = []
        sentence = Cleaner.clean_spaces(sentence)
        for word in sentence.split():
            special_count = len([c for c in word if c in special])
            if special_count < from_size or len(word) != special_count:
                final_words.append(word)
        return ' '.join(final_words)

    @staticmethod
    @accepts(str, special=set)
    @returns(str)
    def remove_isolated_special_chars(sentence,
                                       special=set("+<=>^_\\°²")):
        """Remove isolated special characters.
        """
        final_words = []
        sentence = Cleaner.clean_spaces(sentence)
        for word in sentence.split():
            if len(word) > 1 or word not in special:
                final_words.append(word)
        return ' '.join(final_words)

    @staticmethod
    @accepts(str)
    @returns(List[str])
    def split_parenthesis(sentence):
        """Split the text containing parenthesis ()[]{}| by considering the
        different parenthesis as delimiters.
        """
        for x in list("()[]{}"):
            sentence = sentence.replace(x, '|')
        return [x.strip() for x in sentence.split('|') if len(x.strip())>0]

    @staticmethod
    @accepts(str, Set[str])
    @returns(str)
    def remove_not_good_chars(sentence, good_chars):
        """Replace by a space all characters that are not in good_chars"""
        chars = set(list(sentence))
        for c in chars:
            if c not in good_chars:
                sentence = sentence.replace(c, " ")
        return sentence

    @staticmethod
    @returns(str)
    def remove_special_chars(sentence, special_chars=set("#@[]{}<>=^\\_~")):
        """Replace by a space all given special characters from a list of
        sentences"""
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
        for word in sentence.split():
            if len(set(word).difference(Cleaner.good_chars)) == 0:
                final.append(word)
        return ' '.join(final)

    @staticmethod
    @accepts(str)
    @returns(str)
    def clean_punc(sentence):
        """Clean the punctuation : no space before, space after, remove
        duplicate punctuation except for point. Several points is
        mapped to three points. Note that we try to use as less regex as
        possible because regex are time consuming. """

        # remove duplicated spaces
        sentence = Cleaner.clean_spaces(sentence)
        # remove any space before punctuation
        for c in Cleaner.punc_set:
            sentence = sentence.replace(" "+c, c)
        # remove duplicated punctuation
        for c in Cleaner.punc_set_no_period:
            while c+c in sentence:
                sentence = sentence.replace(c+c, c)
        # replace two or more points by three points
        sentence = re.sub(r"\.{2,}", "...", sentence)
        # replace a weird combination of punctuation by a simple one
        sentence = re.sub(r"[\?\!\.\,\;\:]*\?[\?\!\.\,\;\:]*", "?", sentence)
        sentence = re.sub(r"[\!\.\,\;\:]*\![\!\.\,\;\:]*", "!", sentence)
        sentence = re.sub(r"[\,\;\:]*\.[\,\;\:]*", ".", sentence)
        sentence = re.sub(r"[\,\;\:]*\,[\,\;\:]*", ",", sentence)
        # remove punctuation at the beginning
        while len(sentence) > 0 and sentence[0] in Cleaner.punc_set:
            sentence = sentence[1:]
        # add space after punctuation
        sentence = re.sub(r"\.(?!\.)", ". ", sentence)
        for c in Cleaner.punc_set_no_period:
            sentence = sentence.replace(c, c+" ")
        # remove duplicated space
        sentence = Cleaner.clean_spaces(sentence)
        return sentence

    @staticmethod
    @accepts(str)
    @returns(str)
    def clean_spaces(sentence):
        """Remove duplicated spaces and spaces at the beginning or end of a
        sentence"""
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence

    def add_num_token(sentence):
        """Convert all numbers in a <num> token and isolate the
        token by adding a space before and after.
        """
        sentence = re.sub(r"[0-9][0-9\.\,\'/]*[0-9]", " <num> ", sentence)
        sentence = re.sub(r"\<num\>(\s+\<num\>)+", "<num>", sentence)
        return sentence

    def add_punc_token(sentence):
        """Convert all punctuation signs in a <punc> token and isolate the
        token by adding a space before and after. Make sure to run this after
        add_num_token.
        """
        return re.sub(r"[\.\,\:\;\?\!\]\(\)\"]+", " <punc> ", sentence)
        sentence = re.sub(r"\<num\>(\s+\<punc\>)+", "<punc>", sentence)
        return sentence

    def remove_special_duplication(sentence,
                        special=set("-,?!%&\"\'()/$*+:;<=>[]\\^_{}|~€°²")):
        """Remove the duplication of special characters
        """

        # would do the job but takes slightly more time
        special = r"[\-\,\?\!\%\&\"\'\(\)\/\$\*\+\:\;\<\=\>\[\]\\\^" + \
                   r"\_\{\}\|\~\€\°\²]"
        regex = r"(" + special + r")\1*"
        sentence = re.sub(regex, r"\1", sentence)
        return sentence

        # slightly faster but need to check escaped characters
        # for c in list(special) + [" "]:
        #     if c in sentence:
        #         regex = "\\" + c + "+"
        #         sentence = re.sub(regex, c, sentence)
        # return sentence

    def remove_non_gsw_accent(sentence, exclude={'Ä','Ö','Ü','ä','ö','ü'}):
        """Replace all non gsw accent by the letter without the accent"""
        res = []
        for c in sentence:
            if not c in exclude:
                c = unidecode.unidecode(c)
            res.append(c)
        return ''.join(res)

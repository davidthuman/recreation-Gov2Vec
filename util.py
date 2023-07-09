

class Query():
    """ 
    """
    def __init__(self,word_vocab, gov_vocab, word_embed, gov_embed):
        # Set Vocabs
        self.word_vocab = word_vocab
        self.gov_vocab = gov_vocab
        # Set Embeddings
        self.word_embed = word_embed
        self.gov_embed = gov_embed
        # Init query
        self.query_vec = None

    def set_query(self, query):
        """ Set query for processing

        :param query: query to process, format -> "(word|gov) [(+|-) (word|gov)]*"
        :type query: str
        """
        split_query = query.split(" ")
        processed_query = []
        curr_sign = 1
        curr_term = [0]
        for term in split_query[1:]:

            match term:

                case '+':
                    
                    if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        processed_query.append(self.gov_embed(gov_id) * curr_sign)
                    elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        processed_query.append(self.word_embed(word_id) * curr_sign)
                    else:
                        raise ValueError(f"""Invalid query term has been provide,
                                         {curr_term} is not in word or government vocab.""")
                    curr_sign = 1
                    curr_term = ""

                case '-':

                    if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        processed_query.append(self.gov_embed(gov_id) * curr_sign)
                    elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        processed_query.append(self.word_embed(word_id) * curr_sign)
                    else:
                        raise ValueError(f"""Invalid query term has been provide,
                                         {curr_term} is not in word or government vocab.""")
                    curr_sign = -1
                    curr_term = ""

                case _:

                    if curr_term == "":
                        curr_term = term
                    else:
                        curr_term = curr_term + ' ' + term
        
        # Process last term
        if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        processed_query.append(self.gov_embed(gov_id) * curr_sign)
        elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
            processed_query.append(self.word_embed(word_id) * curr_sign)
        else:
            raise ValueError(f"""Invalid query term has been provide,
                                {curr_term} is not in word or government vocab.""")
        
        # lol, sorry that my code is not DRY, "LeT mE aBtRacT iNTo a fUncTiOn", nah bro
        self.query_vec = sum(processed_query)

    def get_words(self, top=5):
        pass

    def get_govs(self, top=5):
        pass
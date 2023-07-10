
import torch
import tqdm

class Query():
    """ 
    """
    def __init__(self, word_vocab, gov_vocab, word_embed, gov_embed, device):
        # Set Vocabs
        self.word_vocab = word_vocab
        self.gov_vocab = gov_vocab
        # Set Embeddings
        self.word_embed = word_embed
        self.gov_embed = gov_embed
        # Init query
        self.query_terms = []
        self.query_vec = None

        self.device = device

    def set_query(self, query):
        """ Set query for processing

        :param query: query to process, format -> "(word|gov) [(+|-) (word|gov)]*"
        :type query: str
        """
        split_query = query.split(" ")
        processed_query = []
        curr_sign = 1
        curr_term = split_query[0]
        for term in split_query[1:]:

            match term:

                case '+':
                    
                    if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.gov_embed(torch.tensor(gov_id, device=self.device)) * curr_sign)
                    elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.word_embed(torch.tensor(word_id, device=self.device)) * curr_sign)
                    else:
                        raise ValueError(f"""Invalid query term has been provide,
                                         {curr_term} is not in word or government vocab.""")
                    curr_sign = 1
                    curr_term = ""

                case '-':

                    if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.gov_embed(torch.tensor(gov_id, device=self.device)) * curr_sign)
                    elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.word_embed(torch.tensor(word_id, device=self.device)) * curr_sign)
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
            self.query_terms.append(curr_term)
            processed_query.append(self.gov_embed(torch.tensor(gov_id, device=self.device)) * curr_sign)
        elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
            self.query_terms.append(curr_term)
            processed_query.append(self.word_embed(torch.tensor(word_id, device=self.device)) * curr_sign)
        else:
            raise ValueError(f"""Invalid query term has been provide,
                                {curr_term} is not in word or government vocab.""")
        
        # lol, sorry that my code is not DRY, "LeT mE aBtRacT iNTo a fUncTiOn", nah bro
        self.query_vec = sum(processed_query)

    def get_words(self, top=5):
        cos = torch.nn.CosineSimilarity(dim=0)
        rankings = []
        for word, id in tqdm.tqdm(self.word_vocab.word2id.items()):
            # Exclude words that are in the query
            if word in self.query_terms:
                continue
            score = cos(self.word_embed(torch.tensor(id, device=self.device)), self.query_vec).item()
            rankings.append((word, score))
        
        rankings.sort(reverse=True, key=lambda x: x[1])
        return rankings[:top]

    def get_govs(self, top=5):
        cos = torch.nn.CosineSimilarity(dim=0)
        rankings = []
        for gov, id in self.gov_vocab.word2id.items():
            # Exclude words that are in the query
            if gov in self.query_terms:
                continue
            score = cos(self.gov_embed(torch.tensor(id, device=self.device)), self.query_vec).item()
            rankings.append((gov, score))
        
        rankings.sort(reverse=True, key=lambda x: x[1])
        return rankings[:top]
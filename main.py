import string
import random
import time

from matplotlib import pyplot as plt

POPULATION_SIZE = 500
MUTATION_RATE = 0.05
SELECTION_RATE = 0.1
GENERATION_NUM = 500
NPAIRS = 50


# reads the file
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()


# creates a dict that contains the letter/pair as keys and their frequency as values
def read_freq_file(filename):
    freq_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split('\t')
            if len(values) == 2:
                freq, word = values
                freq_dict[word] = float(freq)
    return freq_dict


class GeneticAlg:
    def __init__(self, enc_file):
        self.enc_text = read_file(enc_file)
        self.dict = set(word.lower() for word in read_file('dict.txt').split())
        self.letter_freq = {letter.lower(): freq for letter, freq in read_freq_file('Letter_Freq.txt').items()}
        self.pair_letter_freq = {f"{pair[0].lower()}{pair[1].lower()}": freq for pair, freq in
                                 read_freq_file('Letter2_Freq.txt').items()}
        self.fitness_calls = 0
        self.population = self.init_population()
        self.generations = 0
        self.best_fitness = '-inf'
        self.alephbeit = string.ascii_lowercase
        self.list_of_best_fitness = []
        self.list_generations = []
        self.same_fitness_count = 0
        self.algo_type = 'basic'

    def replace_duplicate_letters(self, child):
        # count the occurrences of each letter
        letter_counts = {}
        for letter in child:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        # replace one of the duplicates with a new letter
        new_s = ''
        missing_letters = [l for l in self.alephbeit if l not in child]
        for letter in child:
            if letter_counts[letter] > 1:
                # Replace one of the duplicates with a new letter
                new_letter = missing_letters.pop()
                new_s += new_letter
                letter_counts[new_letter] = 1
                letter_counts[letter] -= 1

            else:
                new_s += letter
        return new_s

    def init_population(self):
        # Create an empty list to store the (permutation, score) tuples
        perm_fitness_list = []

        # Generate POPULATION_SIZE different permutations of the lowercase alphabet
        for _ in range(POPULATION_SIZE):
            perm = ''.join(random.sample(string.ascii_lowercase, len(string.ascii_lowercase)))
            score = self.individual_fitness(perm)
            perm_fitness_list.append((perm, score))

        return perm_fitness_list

    # encrypt the text according to the permutation
    def encrypt(self, permutation):
        table = str.maketrans(string.ascii_lowercase, permutation)
        return self.enc_text.translate(table)

    # selection
    def selection(self, ):
        # choose the best permutations - means with the highest fitness score
        sorted_d = sorted(self.population, key=lambda x: x[1], reverse=True)
        top_num = int(len(sorted_d) * SELECTION_RATE)
        # top_individuals = sorted_d[:top_num]
        top_individuals = []
        keys = [item[0] for item in top_individuals]
        i = 0
        while i < top_num:
            if sorted_d[i][0] not in keys:
                top_individuals.append(sorted_d[i])
                keys.append(sorted_d[i][0])
                i += 1
            else:
                top_num += 1
                i += 1


        return top_individuals

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        child1 = self.replace_duplicate_letters(child1)
        child2 = self.replace_duplicate_letters(child2)
        return child1, child2

    def mutation(self):
        next_generation = []
        for perm_tuple in self.population:
            perm = perm_tuple[0]
            if random.random() < MUTATION_RATE:
                # generate a mutated permutation by swapping two characters
                permutation = list(perm)
                idx1, idx2 = random.sample(range(len(permutation)), 2)
                permutation[idx1], permutation[idx2] = permutation[idx2], permutation[idx1]
                new_perm = ''.join(permutation)
                next_generation.append((new_perm, self.individual_fitness(new_perm)))
            else:
                next_generation.append((perm, self.individual_fitness(perm)))
        self.population = next_generation

    def individual_fitness(self, perm):
        score = 0
        self.fitness_calls += 1
        dec_text = self.encrypt(perm)
        # iterate over each word in the individual
        for word in dec_text.split():
            # Check if the word exists in the word list and add its length to the score if it does
            if word.lower() in self.dict:
                score += len(word)
        # iterate over each letter and gives the score according to the frequencies in the letter_freq
        for letter in dec_text:
            if letter.isalpha():
                if letter in self.letter_freq:
                    score += self.letter_freq[letter]

        # iterate over each pair of letter and gives the score according to the frequencies in the letter_pair_freq
        for i in range(len(dec_text) - 1):
            letter_pair = dec_text[i:i + 2].lower()
            if letter_pair.isalpha():
                if letter_pair in self.pair_letter_freq:
                    score += self.pair_letter_freq[letter_pair]

        return score

    def set_algo_type(self, algo_type):
        self.algo_type = algo_type

    def darwin_alg(self):
        next_generation = []
        for index in range(POPULATION_SIZE):
            perm = self.population[index][0]
            old_fitness = self.population[index][1]  # saves the old fitness for each permutation
            permutation = list(perm)
            for i in range(NPAIRS):
                # generate a mutated permutation by swapping two characters
                idx1, idx2 = random.sample(range(len(permutation)), 2)
                permutation[idx1], permutation[idx2] = permutation[idx2], permutation[idx1]
            permutation_string = str(''.join(permutation))
            new_fitness = self.individual_fitness(permutation_string)

            if new_fitness > old_fitness:
                next_generation.append((perm, new_fitness))
            else:
                next_generation.append((perm, old_fitness))
        self.population = next_generation

    def lamark_alg(self):
        next_generation = []
        for index in range(POPULATION_SIZE):
            perm = self.population[index][0]
            old_fitness = self.population[index][1]  # saves the old fitness for each permutation
            permutation = list(perm)
            for i in range(NPAIRS):
                # generate a mutated permutation by swapping two characters
                idx1, idx2 = random.sample(range(len(permutation)), 2)
                permutation[idx1], permutation[idx2] = permutation[idx2], permutation[idx1]
            permutation_string = str(''.join(permutation))
            new_fitness = self.individual_fitness(permutation_string)

            if new_fitness > old_fitness:
                next_generation.append((permutation_string, new_fitness))
            else:
                next_generation.append((perm, old_fitness))
        self.population = next_generation

    def run_algorithm(self):
        for i in range(GENERATION_NUM):
            if self.algo_type == 'darwin':
                self.darwin_alg()
            elif self.algo_type == 'lamark':
                self.lamark_alg()
            parents = self.selection()  # list of tuples -(permutation,score)
            keys = [item[0] for item in parents]
            values = [item[1] for item in parents]
            next_generation = []
            j = 0
            while j < int(POPULATION_SIZE / 2):
                parent1 = random.choices(keys, weights=values)[0]
                parent2 = random.choices(keys, weights=values)[0]
                if parent1 == parent2:
                    continue
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.append((child1, 0))
                next_generation.append((child2, 0))
                j += 1
            self.population = next_generation
            self.mutation()
            self.generations += 1
            self.best_fitness = max(self.population, key=lambda x: x[1])[1]
            if self.generations > 1:
                if self.list_of_best_fitness[-1] == self.best_fitness:
                    self.same_fitness_count += 1
                    if self.same_fitness_count > 30:
                        break
                elif self.list_of_best_fitness[-1] != self.best_fitness:
                    self.same_fitness_count = 0
            self.list_generations.append(self.generations)
            self.list_of_best_fitness.append(self.best_fitness)

            print(f"Generation: {self.generations}, Best fitness: {self.best_fitness}")
            print("fitness calls: ", self.fitness_calls)


def create_png(generations, fitness_score):
    """
    Creates a PNG image with a line plot of the number of infected cells over time (generations).

    :param generations: A list of integers representing the number of generations that have passed.
    :param fitness_score: A list of integers representing the fitness score at each generation.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(generations, fitness_score)
    ax.set_xlabel("Generations")
    ax.set_ylabel("fitness score")
    ax.set_title("Number of fitness score Over Time")
    plt.savefig("fitness_score_over_time.png")


def main():
    start_time = time.time()
    file_name = input('Please enter the name of the file: ')
    ga = GeneticAlg(file_name)
    input_string = input("Please enter the algorithm you want to use: ")
    if input_string == "darwin":
        ga.set_algo_type("darwin")
    elif input_string == "lamark":
        ga.set_algo_type("lamark")

    ga.run_algorithm()

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print("running time:", total_time)
    print("the converged generation:", ga.generations - 30)

    create_png(ga.list_generations, ga.list_of_best_fitness)


if __name__ == '__main__':
    main()

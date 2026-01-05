#!/usr/bin/env python3
"""
Dual-Layer Architecture:
1. GA Layer 1: Feature Subset Selection with complexity-aware fitness
2. GA Layer 2: SVM Hyperparameter Optimization with adaptive search
"""

import os
import re
import sys
import json
import random
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, issparse
from scipy.stats import rankdata

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import shuffle

import torch
from transformers import BertTokenizer, BertModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Configuration Constants 
# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Dataset paths
DATA_FILE = "train.tsv"  # LIAR dataset
BERT_MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./genesis_fn_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GA Layer 1: Feature Selection Parameters
GA1_POPULATION_SIZE = 50       # Population size for feature selection GA
GA1_GENERATIONS = 30          # Number of generations
GA1_CROSSOVER_RATE = 0.8      # Probability of crossover
GA1_MUTATION_RATE = 0.05      # Probability of mutation per gene
GA1_TOURNAMENT_SIZE = 3       # Tournament size for selection
GA1_ELITISM_COUNT = 2         # Number of best individuals preserved
GA1_INIT_SELECTION_RATE = 0.3 # Initial feature selection rate

# GA Layer 2: SVM Hyperparameter Optimization
GA2_POPULATION_SIZE = 30      # Population size for hyperparameter GA
GA2_GENERATIONS = 20          # Number of generations
GA2_CROSSOVER_RATE = 0.7      # Crossover probability
GA2_MUTATION_RATE = 0.1       # Mutation probability
GA2_ELITISM_COUNT = 2         # Elitism count
GA2_K_FOLDS = 3               # Cross-validation folds for fitness evaluation

# SVM Hyperparameter Ranges (Paper values)
SVM_C_MIN, SVM_C_MAX = 0.1, 10.0           # Regularization parameter C
SVM_GAMMA_MIN, SVM_GAMMA_MAX = 0.001, 1.0  # RBF kernel gamma
SVM_KERNEL_TYPES = ['linear', 'rbf']       # Kernel types to optimize

# Fitness Function Weights (Balancing performance vs complexity)
F1_WEIGHT = 0.35              # Weight for F1 score (primary metric)
ACCURACY_WEIGHT = 0.20        # Weight for accuracy
PRECISION_WEIGHT = 0.15       # Weight for precision
RECALL_WEIGHT = 0.15          # Weight for recall
COMPLEXITY_WEIGHT = 0.15      # Weight for feature count penalty

# Feature Selection Penalty Parameters
PENALTY_EXPONENT = 0.5        # Exponent for feature count penalty (sqrt)
MIN_FEATURES = 5              # Minimum features to avoid empty selection
MAX_FEATURE_PENALTY = 0.3     # Maximum penalty for too many features

# Text Preprocessing Utilities
def initialize_nltk_resources():
    """Initialize required NLTK resources."""
    resources = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

class TextPreprocessor:
    """Comprehensive text preprocessing for fake news detection."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Additional fake-news specific stopwords
        self.fake_news_stopwords = {
            'breaking', 'exclusive', 'shocking', 'amazing', 'unbelievable',
            'must', 'read', 'share', 'viral', 'secret', 'hidden', 'truth'
        }
        self.stop_words.update(self.fake_news_stopwords)
    
    def preprocess(self, text: str) -> str:
        """Complete text preprocessing pipeline."""
        if not isinstance(text, str):
            text = str(text) if pd.notna(text) else ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Stem and lemmatize
        tokens = [self.stemmer.stem(word) for word in tokens]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove short tokens
        tokens = [word for word in tokens if len(word) > 2]
        
        return ' '.join(tokens)

# Data Loading and Preparation

def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load LIAR dataset and prepare labels."""
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job', 
        'state', 'party', 'barely_true_c', 'false_c', 'half_true_c',
        'mostly_true_c', 'pants_on_fire_c', 'venue'
    ]
    
    df = pd.read_csv(filepath, sep='\t', names=columns)
    
    # Convert labels to binary: true/mostly-true/half-true = 1, others = 0
    def binarize_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).lower().strip()
        if label_str in ['true', 'mostly-true', 'half-true']:
            return 1  # Real news
        else:
            return 0  # Fake news
    
    y = df['label'].apply(binarize_label)
    
    # Remove rows with empty statements
    df = df[df['statement'].notna()]
    y = y[df.index]
    
    return df, y

# Hybrid Feature Engineering (Section 3.3)

class HybridFeatureExtractor:
       
    def __init__(self, use_bert: bool = True, bert_batch_size: int = 16):
        self.use_bert = use_bert
        self.bert_batch_size = bert_batch_size
        self.tfidf_vectorizer = None
        self.onehot_encoder = None
        self.scaler = None
        self.bert_tokenizer = None
        self.bert_model = None
        
    def extract_tfidf_features(self, texts: List[str], max_features: int = 5000) -> csr_matrix:
        """Extract TF-IDF features with n-grams."""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                use_idf=True
            )
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def extract_bert_embeddings(self, texts: List[str]) -> csr_matrix:
        """Extract BERT embeddings with mean pooling."""
        if not self.use_bert:
            return csr_matrix((len(texts), 0))
        
        if self.bert_tokenizer is None or self.bert_model is None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
            self.bert_model.eval()
        
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.bert_batch_size):
                batch_texts = texts[i:i + self.bert_batch_size]
                inputs = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                outputs = self.bert_model(**inputs)
                # Mean pooling of last hidden state
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(batch_embeddings)
        
        X_bert = np.vstack(embeddings)
        
        # Standardize BERT embeddings
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_bert = self.scaler.fit_transform(X_bert)
        else:
            X_bert = self.scaler.transform(X_bert)
        
        return csr_matrix(X_bert)
    
    def extract_metadata_features(self, df: pd.DataFrame) -> csr_matrix:
        """Extract categorical and numerical metadata features."""
        # Categorical features
        categorical_cols = ['subject', 'speaker', 'job', 'state', 'party', 'venue']
        
        if self.onehot_encoder is None:
            self.onehot_encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=True,
                max_categories=50  # Limit categories to avoid explosion
            )
            X_cat = self.onehot_encoder.fit_transform(df[categorical_cols])
        else:
            X_cat = self.onehot_encoder.transform(df[categorical_cols])
        
        # Numerical features (fact-checking history)
        numeric_cols = [
            'barely_true_c', 'false_c', 'half_true_c', 
            'mostly_true_c', 'pants_on_fire_c'
        ]
        
        # Normalize numerical features
        X_num = df[numeric_cols].astype(float).values
        X_num = (X_num - X_num.mean(axis=0)) / (X_num.std(axis=0) + 1e-8)
        
        return hstack([X_cat, csr_matrix(X_num)])
    
    def extract_readability_features(self, texts: List[str]) -> np.ndarray:
        """Extract readability scores (Flesch-Kincaid, etc.)."""
        readability_scores = []
        
        for text in texts:
            # Simple readability metrics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            
            if len(sentences) == 0 or len(words) == 0:
                readability_scores.append([0, 0, 0])
                continue
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            unique_word_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
            
            readability_scores.append([
                avg_sentence_length,
                avg_word_length,
                unique_word_ratio
            ])
        
        return np.array(readability_scores)
    
    def extract_all_features(self, df: pd.DataFrame, texts: List[str]) -> Tuple[csr_matrix, Dict]:
        """Extract complete hybrid feature set."""
        print("  Extracting TF-IDF features...")
        X_tfidf = self.extract_tfidf_features(texts)
        
        print("  Extracting BERT embeddings..." if self.use_bert else "  Skipping BERT...")
        X_bert = self.extract_bert_embeddings(texts)
        
        print("  Extracting metadata features...")
        X_meta = self.extract_metadata_features(df)
        
        print("  Extracting readability features...")
        X_readability = csr_matrix(self.extract_readability_features(texts))
        
        # Combine all features
        X_hybrid = hstack([X_tfidf, X_bert, X_meta, X_readability])
        
        # Feature metadata
        feature_metadata = {
            'tfidf_features': X_tfidf.shape[1],
            'bert_features': X_bert.shape[1],
            'metadata_features': X_meta.shape[1],
            'readability_features': X_readability.shape[1],
            'total_features': X_hybrid.shape[1],
            'feature_types': {
                'tfidf': X_tfidf.shape[1],
                'bert': X_bert.shape[1],
                'categorical': self.onehot_encoder.get_feature_names_out().shape[0] if self.onehot_encoder else 0,
                'numerical': 5,  # 5 fact-checking columns
                'readability': 3
            }
        }
        
        return X_hybrid, feature_metadata

# Genetic Algorithm Layer 1: Feature Selection (Section 4.0.2)


class FeatureSelectionGA:
    """Genetic Algorithm for feature subset selection with detailed fitness evaluation."""
    
    def __init__(self, 
                 pop_size: int = GA1_POPULATION_SIZE,
                 generations: int = GA1_GENERATIONS,
                 crossover_rate: float = GA1_CROSSOVER_RATE,
                 mutation_rate: float = GA1_MUTATION_RATE,
                 tournament_size: int = GA1_TOURNAMENT_SIZE,
                 elitism_count: int = GA1_ELITISM_COUNT):
        
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Fitness evaluation cache to avoid recomputation
        self.fitness_cache = {}
        
    def initialize_population(self, n_features: int) -> np.ndarray:
        """Initialize binary population with diverse feature selection rates."""
        population = []
        
        # Create diverse initial population
        for i in range(self.pop_size):
            if i < self.pop_size // 3:
                # Low feature selection (10-30%)
                selection_rate = random.uniform(0.1, 0.3)
            elif i < 2 * self.pop_size // 3:
                # Medium feature selection (30-60%)
                selection_rate = random.uniform(0.3, 0.6)
            else:
                # High feature selection (60-90%)
                selection_rate = random.uniform(0.6, 0.9)
            
            chromosome = np.random.rand(n_features) < selection_rate
            population.append(chromosome.astype(int))
        
        return np.array(population)
    
    def calculate_fitness(self, 
                         chromosome: np.ndarray, 
                         X: csr_matrix, 
                         y: np.ndarray,
                         cache_key: Optional[str] = None) -> float:
        """
        Comprehensive fitness function with multiple components:
        
        f(S) = w1*F1 + w2*Accuracy + w3*Precision + w4*Recall - w5*ComplexityPenalty
        
        where complexity penalty = sqrt(n_selected) / sqrt(n_total)
        """
        # Check cache first
        if cache_key and cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]
        
        # Get selected feature indices
        selected_idx = np.where(chromosome)[0]
        n_selected = len(selected_idx)
        n_total = X.shape[1]
        
        # Penalty for no features or too few features
        if n_selected < MIN_FEATURES:
            fitness = 0.0
            if cache_key:
                self.fitness_cache[cache_key] = fitness
            return fitness
        
        # Select features
        if issparse(X):
            X_selected = X[:, selected_idx]
        else:
            X_selected = X[:, selected_idx]
        
        try:
            # Train SVM with default parameters for fitness evaluation
            svm = SVC(kernel='linear', C=1.0, random_state=RANDOM_SEED)
            
            # Use stratified 3-fold cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            
            # Calculate multiple metrics
            f1_scores = cross_val_score(svm, X_selected, y, cv=cv, 
                                       scoring='f1', n_jobs=-1)
            accuracy_scores = cross_val_score(svm, X_selected, y, cv=cv,
                                            scoring='accuracy', n_jobs=-1)
            precision_scores = cross_val_score(svm, X_selected, y, cv=cv,
                                             scoring='precision', n_jobs=-1)
            recall_scores = cross_val_score(svm, X_selected, y, cv=cv,
                                          scoring='recall', n_jobs=-1)
            
            # Average scores
            avg_f1 = np.mean(f1_scores)
            avg_accuracy = np.mean(accuracy_scores)
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            
            # Calculate complexity penalty
            # Using sqrt to penalize excessively large feature sets less harshly
            complexity_penalty = (n_selected ** PENALTY_EXPONENT) / (n_total ** PENALTY_EXPONENT)
            
            # Normalize complexity penalty to [0, MAX_FEATURE_PENALTY]
            complexity_penalty = min(complexity_penalty, MAX_FEATURE_PENALTY)
            
            # Combined fitness score
            fitness = (F1_WEIGHT * avg_f1 +
                      ACCURACY_WEIGHT * avg_accuracy +
                      PRECISION_WEIGHT * avg_precision +
                      RECALL_WEIGHT * avg_recall -
                      COMPLEXITY_WEIGHT * complexity_penalty)
            
            # Ensure fitness is non-negative
            fitness = max(fitness, 0.0)
            
        except Exception as e:
            # If model training fails, assign low fitness
            print(f"    Warning: Fitness evaluation failed: {e}")
            fitness = 0.0
        
        # Cache result
        if cache_key:
            self.fitness_cache[cache_key] = fitness
        
        return fitness
    
    def tournament_selection(self, population: np.ndarray, 
                           fitness: np.ndarray) -> np.ndarray:
        """Tournament selection with adaptive pressure."""
        selected = []
        n_individuals = len(population)
        
        for _ in range(self.pop_size):
            # Select tournament participants
            participants = np.random.choice(n_individuals, self.tournament_size, replace=False)
            
            # Find winner (highest fitness)
            winner_idx = participants[np.argmax(fitness[participants])]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def uniform_crossover(self, parent1: np.ndarray, 
                         parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover operator."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Create crossover mask
        mask = np.random.rand(len(parent1)) > 0.5
        
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def adaptive_mutation(self, chromosome: np.ndarray, 
                         generation: int, 
                         max_generation: int) -> np.ndarray:
        """
        Adaptive mutation: Higher mutation rate early, lower later.
        Focuses mutation on less important features.
        """
        mutated = chromosome.copy()
        n_genes = len(chromosome)
        
        # Adaptive mutation rate
        current_mutation_rate = self.mutation_rate * (1.0 - (generation / max_generation))
        current_mutation_rate = max(current_mutation_rate, 0.01)  # Minimum 1%
        
        # Determine which genes to mutate
        for i in range(n_genes):
            if random.random() < current_mutation_rate:
                # Flip the bit
                mutated[i] = 1 - mutated[i]
        
        return mutated
    
    def run(self, X: csr_matrix, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Execute GA feature selection."""
        n_features = X.shape[1]
        
        # Initialize population
        population = self.initialize_population(n_features)
        best_fitness_history = []
        best_chromosome = None
        best_fitness = -np.inf
        
        print(f"  GA Layer 1: Feature Selection")
        print(f"    Population: {self.pop_size}, Generations: {self.generations}")
        print(f"    Total features: {n_features}")
        print(f"    Crossover rate: {self.crossover_rate}, Mutation rate: {self.mutation_rate}")
        
        # Clear cache for new run
        self.fitness_cache = {}
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for i, chromosome in enumerate(population):
                # Create cache key
                cache_key = f"{generation}_{i}_{chromosome.tobytes().hex()[:16]}"
                fitness = self.calculate_fitness(chromosome, X, y, cache_key)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best solution
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = population[gen_best_idx].copy()
            
            best_fitness_history.append(best_fitness)
            
            # Selection
            parents = self.tournament_selection(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Crossover and Mutation
            while len(new_population) < self.pop_size:
                # Select two parents
                idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[idx1], parents[idx2]
                
                # Crossover
                child1, child2 = self.uniform_crossover(parent1, parent2)
                
                # Adaptive mutation
                child1 = self.adaptive_mutation(child1, generation, self.generations)
                child2 = self.adaptive_mutation(child2, generation, self.generations)
                
                new_population.extend([child1, child2])
            
            # Trim if needed
            population = np.array(new_population[:self.pop_size])
            
            # Progress reporting
            if (generation + 1) % 5 == 0 or generation == 0:
                n_selected = np.sum(best_chromosome)
                selection_rate = n_selected / n_features
                print(f"    Gen {generation+1:3d}: "
                      f"Best fitness = {best_fitness:.4f}, "
                      f"Selected = {n_selected:4d} ({selection_rate:.1%})")
        
        # Get final selected features
        selected_indices = np.where(best_chromosome)[0]
        
        # Statistics
        stats = {
            'n_features_total': n_features,
            'n_features_selected': len(selected_indices),
            'selection_rate': len(selected_indices) / n_features,
            'best_fitness': best_fitness,
            'fitness_history': best_fitness_history,
            'final_chromosome': best_chromosome.tolist(),
            'ga_parameters': {
                'pop_size': self.pop_size,
                'generations': self.generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'tournament_size': self.tournament_size,
                'elitism_count': self.elitism_count
            }
        }
        
        print(f"  Final selection: {stats['n_features_selected']} features "
              f"({stats['selection_rate']:.1%})")
        
        return selected_indices, stats

# ----------------------------------------------------------------------
# Genetic Algorithm Layer 2: SVM Hyperparameter Optimization (Section 4.0.3)
# ----------------------------------------------------------------------

class SVMHyperparameterGA:
    """Genetic Algorithm for SVM hyperparameter optimization."""
    
    def __init__(self,
                 pop_size: int = GA2_POPULATION_SIZE,
                 generations: int = GA2_GENERATIONS,
                 crossover_rate: float = GA2_CROSSOVER_RATE,
                 mutation_rate: float = GA2_MUTATION_RATE,
                 elitism_count: int = GA2_ELITISM_COUNT):
        
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        
        # Hyperparameter bounds
        self.c_bounds = (SVM_C_MIN, SVM_C_MAX)
        self.gamma_bounds = (SVM_GAMMA_MIN, SVM_GAMMA_MAX)
        self.kernel_types = SVM_KERNEL_TYPES
        
    def encode_individual(self, C: float, gamma: float, kernel: str) -> np.ndarray:
        """Encode hyperparameters into a normalized chromosome [0, 1]."""
        # Normalize C (log scale)
        C_norm = (np.log10(C) - np.log10(self.c_bounds[0])) / \
                (np.log10(self.c_bounds[1]) - np.log10(self.c_bounds[0]))
        
        # Normalize gamma (log scale)
        gamma_norm = (np.log10(gamma) - np.log10(self.gamma_bounds[0])) / \
                    (np.log10(self.gamma_bounds[1]) - np.log10(self.gamma_bounds[0]))
        
        # Encode kernel type: linear=0.0, rbf=1.0
        kernel_norm = 0.0 if kernel == 'linear' else 1.0
        
        return np.array([C_norm, gamma_norm, kernel_norm])
    
    def decode_individual(self, chromosome: np.ndarray) -> Tuple[float, float, str]:
        """Decode chromosome to hyperparameters."""
        # Ensure values are in [0, 1] range
        C_norm = np.clip(chromosome[0], 0.0, 1.0)
        gamma_norm = np.clip(chromosome[1], 0.0, 1.0)
        kernel_norm = chromosome[2]
        
        # Convert back to actual values (log scale)
        C = 10 ** (np.log10(self.c_bounds[0]) + 
                  C_norm * (np.log10(self.c_bounds[1]) - np.log10(self.c_bounds[0])))
        
        gamma = 10 ** (np.log10(self.gamma_bounds[0]) + 
                      gamma_norm * (np.log10(self.gamma_bounds[1]) - np.log10(self.gamma_bounds[0])))
        
        # Decode kernel type
        kernel = 'linear' if kernel_norm < 0.5 else 'rbf'
        
        return C, gamma, kernel
    
    def initialize_population(self) -> np.ndarray:
        """Initialize population with diverse hyperparameters."""
        population = []
        
        for i in range(self.pop_size):
            if i < self.pop_size // 3:
                # Focus on linear kernel with moderate C
                C = 10 ** np.random.uniform(np.log10(0.5), np.log10(2.0))
                gamma = 10 ** np.random.uniform(np.log10(self.gamma_bounds[0]), 
                                               np.log10(self.gamma_bounds[1]))
                kernel = 'linear'
            elif i < 2 * self.pop_size // 3:
                # Focus on RBF kernel
                C = 10 ** np.random.uniform(np.log10(self.c_bounds[0]), 
                                           np.log10(self.c_bounds[1]))
                gamma = 10 ** np.random.uniform(np.log10(0.01), np.log10(0.1))
                kernel = 'rbf'
            else:
                # Random exploration
                C = 10 ** np.random.uniform(np.log10(self.c_bounds[0]), 
                                           np.log10(self.c_bounds[1]))
                gamma = 10 ** np.random.uniform(np.log10(self.gamma_bounds[0]), 
                                               np.log10(self.gamma_bounds[1]))
                kernel = random.choice(self.kernel_types)
            
            individual = self.encode_individual(C, gamma, kernel)
            population.append(individual)
        
        return np.array(population)
    
    def evaluate_fitness(self, chromosome: np.ndarray, 
                        X: csr_matrix, y: np.ndarray) -> float:
        """Evaluate fitness using SVM with given hyperparameters."""
        C, gamma, kernel = self.decode_individual(chromosome)
        
        try:
            if kernel == 'linear':
                svm = SVC(C=C, kernel='linear', random_state=RANDOM_SEED)
            else:  # rbf
                svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=RANDOM_SEED)
            
            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=GA2_K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            
            # Use F1 score as fitness (primary metric in paper)
            scores = cross_val_score(svm, X, y, cv=cv, scoring='f1', n_jobs=-1)
            fitness = np.mean(scores)
            
            # Add small penalty for extreme parameter values (regularization)
            # This encourages moderate, generalizable parameters
            c_penalty = 0.01 * abs(np.log10(C / 1.0))  # Penalize far from C=1
            gamma_penalty = 0.01 * abs(np.log10(gamma / 0.1))  # Penalize far from gamma=0.1
            
            fitness = fitness - c_penalty - gamma_penalty
            
        except Exception as e:
            print(f"      Warning: Hyperparameter evaluation failed: {e}")
            fitness = 0.0
        
        return max(fitness, 0.0)
    
    def arithmetic_crossover(self, parent1: np.ndarray, 
                           parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Arithmetic crossover for continuous parameters."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Random blending factor
        alpha = random.random()
        
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        
        return child1, child2
    
    def gaussian_mutation(self, chromosome: np.ndarray, 
                         generation: int, 
                         max_generation: int) -> np.ndarray:
        """Gaussian mutation with decaying strength."""
        mutated = chromosome.copy()
        
        # Adaptive mutation strength
        mutation_strength = 0.1 * (1.0 - (generation / max_generation))
        mutation_strength = max(mutation_strength, 0.01)
        
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, mutation_strength)
                mutated[i] += noise
                
                # Clamp to [0, 1] range
                mutated[i] = np.clip(mutated[i], 0.0, 1.0)
        
        return mutated
    
    def run(self, X: csr_matrix, y: np.ndarray) -> Dict:
        """Execute GA hyperparameter optimization."""
        population = self.initialize_population()
        best_fitness_history = []
        best_chromosome = None
        best_fitness = -np.inf
        best_params_history = []
        
        print(f"\n  GA Layer 2: SVM Hyperparameter Optimization")
        print(f"    Population: {self.pop_size}, Generations: {self.generations}")
        print(f"    C range: [{self.c_bounds[0]}, {self.c_bounds[1]}]")
        print(f"    Gamma range: [{self.gamma_bounds[0]}, {self.gamma_bounds[1]}]")
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for chromosome in population:
                fitness = self.evaluate_fitness(chromosome, X, y)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best solution
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = population[gen_best_idx].copy()
            
            best_fitness_history.append(best_fitness)
            
            # Decode and store best parameters
            best_C, best_gamma, best_kernel = self.decode_individual(best_chromosome)
            best_params_history.append({
                'generation': generation,
                'C': best_C,
                'gamma': best_gamma,
                'kernel': best_kernel,
                'fitness': best_fitness
            })
            
            # Selection (tournament)
            selected = []
            n_individuals = len(population)
            
            for _ in range(self.pop_size):
                participants = np.random.choice(n_individuals, 3, replace=False)
                winner_idx = participants[np.argmax(fitness_scores[participants])]
                selected.append(population[winner_idx])
            
            selected = np.array(selected)
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Crossover and mutation
            while len(new_population) < self.pop_size:
                idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
                child1, child2 = self.arithmetic_crossover(selected[idx1], selected[idx2])
                
                child1 = self.gaussian_mutation(child1, generation, self.generations)
                child2 = self.gaussian_mutation(child2, generation, self.generations)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.pop_size])
            
            # Progress reporting
            if (generation + 1) % 5 == 0 or generation == 0:
                print(f"    Gen {generation+1:3d}: "
                      f"Best fitness = {best_fitness:.4f}, "
                      f"C = {best_C:.3f}, gamma = {best_gamma:.4f}, kernel = {best_kernel}")
        
        # Final best parameters
        best_C, best_gamma, best_kernel = self.decode_individual(best_chromosome)
        
        stats = {
            'best_C': best_C,
            'best_gamma': best_gamma,
            'best_kernel': best_kernel,
            'best_fitness': best_fitness,
            'fitness_history': best_fitness_history,
            'parameters_history': best_params_history,
            'ga_parameters': {
                'pop_size': self.pop_size,
                'generations': self.generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'elitism_count': self.elitism_count
            }
        }
        
        print(f"  Final parameters: C={best_C:.3f}, gamma={best_gamma:.4f}, "
              f"kernel={best_kernel}, fitness={best_fitness:.4f}")
        
        return stats

# ----------------------------------------------------------------------
# Main Genesis-FN Pipeline
# ----------------------------------------------------------------------

class GenesisFNPipeline:
    """Complete Genesis-FN pipeline with dual-layer GA optimization."""
    
    def __init__(self, 
                 use_bert: bool = True,
                 save_artifacts: bool = True,
                 verbose: bool = True):
        
        self.use_bert = use_bert
        self.save_artifacts = save_artifacts
        self.verbose = verbose
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = HybridFeatureExtractor(use_bert=use_bert)
        self.feature_selector = FeatureSelectionGA()
        self.hyperparam_optimizer = SVMHyperparameterGA()
        
        # Results storage
        self.results = {}
        self.selected_features = None
        self.best_hyperparams = None
        self.final_model = None
        self.feature_metadata = None
        
    def log(self, message: str, level: str = "INFO"):
        """Logging utility."""
        if self.verbose:
            prefix = {
                "INFO": "[INFO]",
                "WARNING": "[WARNING]",
                "ERROR": "[ERROR]",
                "SUCCESS": "[SUCCESS]"
            }.get(level, "[INFO]")
            print(f"{prefix} {message}")
    
    def run(self, data_path: str) -> Dict:
        """Execute complete Genesis-FN pipeline."""
        self.log("=" * 70)
        self.log("GENESIS-FN: Evolutionary Fake News Detection Pipeline")
        self.log("=" * 70)
        
        # Step 1: Load and prepare data
        self.log("\n1. Loading and preparing data...")
        df, y = load_and_prepare_data(data_path)
        self.log(f"   Loaded {len(df)} samples, {y.sum()} real news, {len(y)-y.sum()} fake news")
        
        # Step 2: Preprocess text
        self.log("\n2. Preprocessing text...")
        texts = df['statement'].apply(self.text_preprocessor.preprocess)
        
        # Step 3: Create hybrid features
        self.log("\n3. Creating hybrid feature set...")
        X_hybrid, self.feature_metadata = self.feature_extractor.extract_all_features(df, texts.tolist())
        self.log(f"   Created {self.feature_metadata['total_features']} hybrid features")
        self.log(f"   Feature breakdown: TF-IDF={self.feature_metadata['feature_types']['tfidf']}, "
                f"BERT={self.feature_metadata['feature_types']['bert']}, "
                f"Categorical={self.feature_metadata['feature_types']['categorical']}, "
                f"Numerical=5, Readability=3")
        
        # Step 4: Split data
        self.log("\n4. Splitting data into train/validation/test sets...")
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_hybrid, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_temp
        )
        
        self.log(f"   Training set: {X_train.shape[0]} samples")
        self.log(f"   Validation set: {X_val.shape[0]} samples")
        self.log(f"   Test set: {X_test.shape[0]} samples")
        
        # Step 5: GA Layer 1 - Feature Selection
        self.log("\n5. Genetic Algorithm Layer 1: Feature Selection")
        selected_indices, fs_stats = self.feature_selector.run(X_train, y_train)
        self.selected_features = selected_indices
        
        # Apply feature selection to all sets
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        self.log(f"   Selected {len(selected_indices)} features "
                f"({fs_stats['selection_rate']:.1%} of total)")
        
        # Step 6: GA Layer 2 - SVM Hyperparameter Optimization
        self.log("\n6. Genetic Algorithm Layer 2: SVM Hyperparameter Optimization")
        hp_stats = self.hyperparam_optimizer.run(X_train_selected, y_train)
        self.best_hyperparams = {
            'C': hp_stats['best_C'],
            'gamma': hp_stats['best_gamma'],
            'kernel': hp_stats['best_kernel']
        }
        
        # Step 7: Train final model with optimized parameters
        self.log("\n7. Training final optimized model...")
        if self.best_hyperparams['kernel'] == 'linear':
            self.final_model = SVC(
                C=self.best_hyperparams['C'],
                kernel='linear',
                random_state=RANDOM_SEED,
                probability=True
            )
        else:  # rbf
            self.final_model = SVC(
                C=self.best_hyperparams['C'],
                gamma=self.best_hyperparams['gamma'],
                kernel='rbf',
                random_state=RANDOM_SEED,
                probability=True
            )
        
        # Train on combined train+validation for final model
        X_final_train = hstack([X_train_selected, X_val_selected])
        y_final_train = np.concatenate([y_train, y_val])
        
        self.final_model.fit(X_final_train, y_final_train)
        
        # Step 8: Evaluate model on test set
        self.log("\n8. Evaluating final model on test set...")
        y_pred = self.final_model.predict(X_test_selected)
        y_pred_proba = self.final_model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'test_set_size': len(y_test),
            'predictions': {
                'true': y_test.tolist(),
                'predicted': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
        }
        
        # Step 9: Store results
        self.results = {
            'dataset_info': {
                'total_samples': len(df),
                'real_news_count': int(y.sum()),
                'fake_news_count': int(len(y) - y.sum()),
                'data_split': {
                    'train': X_train.shape[0],
                    'validation': X_val.shape[0],
                    'test': X_test.shape[0]
                }
            },
            'feature_info': self.feature_metadata,
            'feature_selection': {
                'selected_indices': selected_indices.tolist(),
                'selected_count': len(selected_indices),
                'stats': fs_stats
            },
            'hyperparameter_optimization': {
                'best_parameters': self.best_hyperparams,
                'stats': hp_stats
            },
            'model_performance': metrics,
            'pipeline_config': {
                'use_bert': self.use_bert,
                'random_seed': RANDOM_SEED,
                'ga1_parameters': fs_stats['ga_parameters'],
                'ga2_parameters': hp_stats['ga_parameters']
            }
        }
        
        # Step 10: Save artifacts if requested
        if self.save_artifacts:
            self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save all results to files."""
        import pickle
        import time
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = os.path.join(OUTPUT_DIR, f'genesis_fn_results_{timestamp}.json')
        
        def convert_for_json(obj):
            """Convert numpy/pandas objects to JSON-serializable format."""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert_for_json(self.results), f, indent=2)
        
        # Save model
        model_file = os.path.join(OUTPUT_DIR, f'genesis_fn_model_{timestamp}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': self.final_model,
                'selected_features': self.selected_features,
                'hyperparameters': self.best_hyperparams,
                'feature_metadata': self.feature_metadata
            }, f)
        
        # Save summary report
        summary_file = os.path.join(OUTPUT_DIR, f'genesis_fn_summary_{timestamp}.txt')
        self.create_summary_report(summary_file)
        
        self.log(f"\nResults saved:")
        self.log(f"  Detailed results: {results_file}")
        self.log(f"  Trained model: {model_file}")
        self.log(f"  Summary report: {summary_file}")
    
    def create_summary_report(self, filename: str):
        """Create human-readable summary report."""
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("GENESIS-FN: COMPLETE SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {self.results['dataset_info']['total_samples']}\n")
            f.write(f"Real news: {self.results['dataset_info']['real_news_count']}\n")
            f.write(f"Fake news: {self.results['dataset_info']['fake_news_count']}\n")
            f.write(f"Train/Val/Test split: {self.results['dataset_info']['data_split']['train']}/"
                   f"{self.results['dataset_info']['data_split']['validation']}/"
                   f"{self.results['dataset_info']['data_split']['test']}\n\n")
            
            f.write("2. FEATURE ENGINEERING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total features created: {self.results['feature_info']['total_features']}\n")
            f.write(f"TF-IDF features: {self.results['feature_info']['feature_types']['tfidf']}\n")
            f.write(f"BERT features: {self.results['feature_info']['feature_types']['bert']}\n")
            f.write(f"Categorical features: {self.results['feature_info']['feature_types']['categorical']}\n")
            f.write(f"Numerical features: 5\n")
            f.write(f"Readability features: 3\n\n")
            
            f.write("3. FEATURE SELECTION (GA Layer 1)\n")
            f.write("-" * 40 + "\n")
            fs = self.results['feature_selection']
            f.write(f"Selected features: {fs['selected_count']} "
                   f"({fs['selected_count']/self.results['feature_info']['total_features']:.1%})\n")
            f.write(f"Best fitness: {fs['stats']['best_fitness']:.4f}\n")
            f.write(f"GA Parameters: Population={fs['stats']['ga_parameters']['pop_size']}, "
                   f"Generations={fs['stats']['ga_parameters']['generations']}\n\n")
            
            f.write("4. HYPERPARAMETER OPTIMIZATION (GA Layer 2)\n")
            f.write("-" * 40 + "\n")
            hp = self.results['hyperparameter_optimization']
            f.write(f"Optimal kernel: {hp['best_parameters']['kernel']}\n")
            f.write(f"Optimal C: {hp['best_parameters']['C']:.4f}\n")
            if hp['best_parameters']['kernel'] == 'rbf':
                f.write(f"Optimal gamma: {hp['best_parameters']['gamma']:.4f}\n")
            f.write(f"Best fitness: {hp['stats']['best_fitness']:.4f}\n")
            f.write(f"GA Parameters: Population={hp['stats']['ga_parameters']['pop_size']}, "
                   f"Generations={hp['stats']['ga_parameters']['generations']}\n\n")
            
            f.write("5. FINAL MODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            metrics = self.results['model_performance']
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"Test set size: {metrics['test_set_size']}\n\n")
            
            f.write("6. PIPELINE CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            config = self.results['pipeline_config']
            f.write(f"BERT used: {config['use_bert']}\n")
            f.write(f"Random seed: {config['random_seed']}\n")
            f.write(f"Fitness weights: F1={F1_WEIGHT}, Accuracy={ACCURACY_WEIGHT}, "
                   f"Precision={PRECISION_WEIGHT}, Recall={RECALL_WEIGHT}, "
                   f"Complexity={COMPLEXITY_WEIGHT}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
    
    def print_final_results(self):
        """Print final results to console."""
        if not self.results:
            self.log("No results available. Run the pipeline first.", "ERROR")
            return
        
        metrics = self.results['model_performance']
        hp = self.results['hyperparameter_optimization']['best_parameters']
        fs = self.results['feature_selection']
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"\n⚙️  OPTIMIZED HYPERPARAMETERS:")
        print(f"   Kernel: {hp['kernel']}")
        print(f"   C:      {hp['C']:.4f}")
        if hp['kernel'] == 'rbf':
            print(f"   Gamma:  {hp['gamma']:.4f}")
        
        print(f"\n🔍 FEATURE SELECTION:")
        total_features = self.results['feature_info']['total_features']
        print(f"   Selected: {fs['selected_count']} out of {total_features} features "
              f"({fs['selected_count']/total_features:.1%})")
        
        print(f"\n🧬 GENETIC ALGORITHM STATISTICS:")
        fs_stats = self.results['feature_selection']['stats']
        hp_stats = self.results['hyperparameter_optimization']['stats']
        print(f"   GA1 Best Fitness: {fs_stats['best_fitness']:.4f}")
        print(f"   GA2 Best Fitness: {hp_stats['best_fitness']:.4f}")
        
        print("\n" + "=" * 70)

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Genesis-FN: Complete Evolutionary Fake News Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data_path train.tsv                    # Run with LIAR dataset
  %(prog)s --no_bert                                # Run without BERT (faster)
  %(prog)s --ga1_generations 20 --ga2_generations 15  # Custom generation counts
        
Paper Reference:
  Genesis-FN: "Hybrid Feature-Based Fake News Detection with Genesis-FN: 
  An Evolutionary Optimization Approach"
        """
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to LIAR dataset TSV file (required)'
    )
    parser.add_argument(
        '--no_bert',
        action='store_true',
        help='Disable BERT embeddings (faster training)'
    )
    parser.add_argument(
        '--ga1_generations',
        type=int,
        default=GA1_GENERATIONS,
        help=f'Number of generations for GA Layer 1 (default: {GA1_GENERATIONS})'
    )
    parser.add_argument(
        '--ga2_generations',
        type=int,
        default=GA2_GENERATIONS,
        help=f'Number of generations for GA Layer 2 (default: {GA2_GENERATIONS})'
    )
    parser.add_argument(
        '--ga1_pop_size',
        type=int,
        default=GA1_POPULATION_SIZE,
        help=f'Population size for GA Layer 1 (default: {GA1_POPULATION_SIZE})'
    )
    parser.add_argument(
        '--ga2_pop_size',
        type=int,
        default=GA2_POPULATION_SIZE,
        help=f'Population size for GA Layer 2 (default: {GA2_POPULATION_SIZE})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for results (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbose output'
    )
    
    args = parser.parse_args()
    
    # Check for required arguments
    if args.data_path is None:
        print("ERROR: --data_path argument is required")
        print("Please provide path to LIAR dataset TSV file")
        sys.exit(1)
    
    # Update output directory
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize NLTK resources
    initialize_nltk_resources()
    
    try:
        # Create and run pipeline
        pipeline = GenesisFNPipeline(
            use_bert=not args.no_bert,
            save_artifacts=True,
            verbose=not args.quiet
        )
        
        # Update GA parameters if provided
        if args.ga1_generations != GA1_GENERATIONS:
            pipeline.feature_selector.generations = args.ga1_generations
        if args.ga2_generations != GA2_GENERATIONS:
            pipeline.hyperparam_optimizer.generations = args.ga2_generations
        if args.ga1_pop_size != GA1_POPULATION_SIZE:
            pipeline.feature_selector.pop_size = args.ga1_pop_size
        if args.ga2_pop_size != GA2_POPULATION_SIZE:
            pipeline.hyperparam_optimizer.pop_size = args.ga2_pop_size
        
        # Run the pipeline
        results = pipeline.run(args.data_path)
        
        # Print final results
        pipeline.print_final_results()
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found: {e}")
        print("Please check the --data_path argument")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

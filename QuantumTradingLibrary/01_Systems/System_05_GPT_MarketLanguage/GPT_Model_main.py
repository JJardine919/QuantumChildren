import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mnemonic import Mnemonic
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

# Конвертер цен в бинарный код
class PriceToBinaryConverter:
    def __init__(self, sequence_length: int = 32):
        self.sequence_length = sequence_length

    def convert_prices_to_binary(self, prices: pd.Series) -> List[str]:
        binary_sequence = []
        for i in range(1, len(prices)):
            binary_digit = '1' if prices.iloc[i] > prices.iloc[i-1] else '0'
            binary_sequence.append(binary_digit)
        return binary_sequence

    def get_binary_chunks(self, binary_sequence: List[str]) -> List[str]:
        chunks = []
        for i in range(0, len(binary_sequence), self.sequence_length):
            chunk = ''.join(binary_sequence[i:i + self.sequence_length])
            if len(chunk) < self.sequence_length:
                chunk = chunk.ljust(self.sequence_length, '0')
            chunks.append(chunk)
        return chunks

# Конвертер BIP39
class BIP39Converter:
    def __init__(self):
        self.mnemo = Mnemonic('english')
        self.wordlist = self.mnemo.wordlist
        self.binary_to_word = {format(i, '011b'): word for i, word in enumerate(self.wordlist)}
        self.word_to_binary = {word: format(i, '011b') for i, word in enumerate(self.wordlist)}
        self.vocab_size = len(self.wordlist)

    def binary_to_bip39(self, binary_sequence: str) -> List[str]:
        words = []
        for i in range(0, len(binary_sequence), 11):
            binary_chunk = binary_sequence[i:i+11]
            if len(binary_chunk) == 11:
                word = self.binary_to_word.get(binary_chunk, 'unknown')
                words.append(word)
        return words

    def bip39_to_binary(self, words: List[str]) -> str:
        binary_sequence = ''
        for word in words:
            binary_sequence += self.word_to_binary.get(word, '0' * 11)
        return binary_sequence

# Датасет для PyTorch
class PriceDataset(Dataset):
    def __init__(self, sequences: List[List[int]], seq_length: int):
        self.sequences = sequences
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return input_seq, target_seq

# Трансформер для прогнозирования цен
class PriceTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Embedding(1024, d_model),
            nn.Dropout(0.1)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        B, L = src.shape
        pos = torch.arange(0, L, device=src.device).unsqueeze(0).repeat(B, 1)
        src = self.embedding(src) * np.sqrt(self.d_model)
        pos_encoding = self.pos_encoder(pos)
        src = src + pos_encoding
        transformer_out = self.transformer_encoder(src)
        output = self.fc_out(transformer_out)
        return output

# Основная модель, объединяющая все компоненты
class PriceLanguageModel:
    def __init__(self, sequence_length: int = 32):
        self.price_converter = PriceToBinaryConverter(sequence_length)
        self.bip39_converter = BIP39Converter()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        binary_sequence = self.price_converter.convert_prices_to_binary(df['close'])
        binary_chunks = self.price_converter.get_binary_chunks(binary_sequence)
        sequences = []
        for chunk in binary_chunks:
            words = self.bip39_converter.binary_to_bip39(chunk)
            indices = [self.bip39_converter.wordlist.index(word) for word in words]
            sequences.append(indices)
        return sequences

    def train_model(self, sequences: List[List[int]], batch_size: int = 32, epochs: int = 10, learning_rate: float = 0.001):
        self.model = PriceTransformer(vocab_size=self.bip39_converter.vocab_size).to(self.device)
        dataset = PriceDataset(sequences, len(sequences[0]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            print(f"\nStarting epoch {epoch+1}/{epochs}")
            total_loss = 0
            all_targets = []
            all_predictions = []

            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                optimizer.zero_grad()
                output = self.model(input_seq)

                loss = criterion(output.view(-1, self.bip39_converter.vocab_size), target_seq.view(-1))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                with torch.no_grad():
                    predictions = torch.argmax(output, dim=-1)
                    all_targets.extend(target_seq.view(-1).cpu().numpy())
                    all_predictions.extend(predictions.view(-1).cpu().numpy())

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

            accuracy = accuracy_score(all_targets, all_predictions)
            print(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}')

    def generate_sequence(self, prompt: List[str], max_length: int = 50, temperature: float = 1.0) -> List[str]:
        self.model.eval()
        prompt_indices = [self.bip39_converter.wordlist.index(word) for word in prompt]
        prompt_tensor = torch.tensor(prompt_indices).unsqueeze(0).to(self.device)
        generated_indices = prompt_indices.copy()

        with torch.no_grad():
            for i in range(max_length - len(prompt)):
                output = self.model(prompt_tensor)
                next_token_logits = output[0, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1).item()

                generated_indices.append(next_token)
                prompt_tensor = torch.tensor(generated_indices).unsqueeze(0).to(self.device)

        generated_words = [self.bip39_converter.wordlist[idx] for idx in generated_indices]
        return generated_words

def main():
    if not mt5.initialize():
        print("Error initializing MT5")
        return

    symbol = "USDJPY"  # Замените на нужный символ
    timeframe = mt5.TIMEFRAME_H1  # Замените на нужный таймфрейм
    start_date = pd.Timestamp('2021-01-01')  # Заменить на нужную дату начала
    end_date = pd.Timestamp('2024-02-04')    # Заменить на нужную дату конца

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    price_model = PriceLanguageModel(sequence_length=32)
    sequences = price_model.prepare_data(df)

    price_model.train_model(sequences, epochs=10)  # Заменить на нужное количество эпох

    # Генерация последовательности
    last_sequence = sequences[-1]
    last_words = [price_model.bip39_converter.wordlist[idx] for idx in last_sequence[:5]]
    predicted_sequence = price_model.generate_sequence(last_words)

    # Преобразование обратно в бинарный код и вывод
    binary_sequence = price_model.bip39_converter.bip39_to_binary(predicted_sequence)
    print("Predicted price movements (1=up, 0=down):", binary_sequence)

    # Расчет общей точности (Accuracy)
    all_targets = []
    all_predictions = []

    # Получаем все input_seq и target_seq из sequences
    for seq in sequences:
        input_seq = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0).to(price_model.device)
        target_seq = torch.tensor(seq[1:], dtype=torch.long).to(price_model.device)

        with torch.no_grad():
            output = price_model.model(input_seq)
            predictions = torch.argmax(output, dim=-1)
            all_targets.extend(target_seq.view(-1).cpu().numpy())
            all_predictions.extend(predictions.view(-1).cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Total Accuracy: {accuracy:.4f}")

    # Прогноз тренда
    ones_count = binary_sequence.count('1')
    total_count = len(binary_sequence)
    trend_percentage = (ones_count / total_count) * 100
    print(f"Trend prediction: {trend_percentage:.2f}% (probability of '1' - up)")

    mt5.shutdown()

if __name__ == "__main__":
    main()

"""
Script para remover linhas com dados sensíveis do arquivo mensagens.csv
"""
import re
import sys
import io

# Configurar encoding UTF-8 para stdout (necessário no Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def contains_sensitive_data(text):
    """Verifica se o texto contém dados sensíveis"""

    # Padrões de CPF (com e sem pontuação)
    cpf_patterns = [
        r'\d{3}\.\d{3}\.\d{3}-\d{2}',  # 000.000.000-00
    ]

    # Padrões de RG (com e sem pontuação)
    rg_patterns = [
        r'\d{2}\.\d{3}\.\d{3}-\d{1,2}',  # 00.000.000-0
        r'\d{8,12}-?\d{0,2}[xX]?',        # RG variável
        r'RG\s*\d{6,}',                   # RG seguido de 6+ dígitos
        r'\brg\b[\s:]+\d{6,}',            # "rg" com 6+ dígitos
    ]

    # Padrões de telefone celular (11 dígitos começando com 9)
    phone_patterns = [
        r'\d{11}',                       # 11999999999
        r'\d{9}-\d{4}',                  # 99999-9999
        r'\(\d{2}\)\s*\d{5}-\d{4}',      # (11) 99999-9999
        r'\d{2}\s*\d{5}-\d{4}',          # 11 99999-9999
    ]

    # Padrões de email
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'

    # Padrões de endereço
    address_patterns = [
        r'Rua\s+\w+.*\d+,.*ap\s*\d+',    # Rua ... 123, ap 456
        r'\d{5}-\d{3}',                  # CEP
    ]

    # Padrões de conta bancária
    bank_patterns = [
        r'\d{10,12}-\d{1}',              # Conta corrente
    ]

    # Padrões de CRECI
    creci_pattern = r'CRECISP\s*\d+'

    # Padrões de passaporte
    passport_pattern = r'passaporte\s*\w+'

    # Padrões de placa de veículo (LLL-NNNN)
    plate_pattern = r'[A-Z]{3}\s*-\s*\d{4}'

    all_patterns = (
        cpf_patterns + rg_patterns + phone_patterns +
        [email_pattern] + address_patterns + bank_patterns +
        [creci_pattern, passport_pattern, plate_pattern]
    )

    for pattern in all_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Verificações adicionais para evitar falsos positivos
            if 'Protocolo:' in text or 'CNPJ' in text:
                continue  # Ignorar protocolos e CNPJ
            return True

    return False

def main():
    input_file = 'mensagens.csv'
    output_file = 'mensagens_clean.csv'

    # Ler o arquivo
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    sensitive_count = 0
    clean_lines = []

    for i, line in enumerate(lines, 1):
        if contains_sensitive_data(line):
            sensitive_count += 1
            # Truncar com cuidado para evitar problemas de encoding
            try:
                preview = line[:80].replace('\n', '').replace('\r', '')
                print(f"Removendo linha {i}: {preview}")
            except:
                print(f"Removendo linha {i}")
        else:
            clean_lines.append(line)

    # Escrever o arquivo limpo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(clean_lines)

    print(f"\n--- Resumo ---")
    print(f"Total de linhas: {total_lines}")
    print(f"Linhas removidas (com dados sensíveis): {sensitive_count}")
    print(f"Linhas mantidas (limpas): {total_lines - sensitive_count}")
    print(f"\nArquivo salvo como: {output_file}")

if __name__ == '__main__':
    main()
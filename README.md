# API Inteligente de Contagem de Ábaco

Desenvolvimento de uma **API Python** voltada à **contagem automática de miçangas em ábacos industriais**, utilizando **visão computacional** (*OpenCV*) e **agrupamento de dados** (*DBSCAN*).
O sistema é capaz de identificar cores, posições e calcular automaticamente os valores representados em **centenas**, **dezenas** e **unidades**.

---

## 📚 Sumário

* [💡 Sobre o Projeto](#-sobre-o-projeto)
* [⚙️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
* [🧩 Como Funciona](#-como-funciona)
* [🧰 Como Executar](#-como-executar)
* [📂 Estrutura do Projeto](#-estrutura-do-projeto)
* [👩‍💻 Autores](#-autores)

---

## 💡 Sobre o Projeto

O **IARA API de Contagem de Ábaco** foi desenvolvido como parte do ecossistema **IARA Tech**, com o objetivo de **automatizar o registro e o monitoramento de ábacos industriais**.

A aplicação utiliza algoritmos de **inteligência artificial e visão computacional** para processar imagens, detectar miçangas e determinar os valores correspondentes, promovendo **eficiência, padronização e precisão** no controle de produção.

---

## ⚙️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Bibliotecas Principais:** OpenCV, NumPy, scikit-learn, Matplotlib
* **Framework:** FastAPI
* **Outros:** GitHub Actions (CI/CD)

---

## 💡 Como Funciona

1. O sistema recebe uma imagem do ábaco industrial.
2. As miçangas coloridas são detectadas e isoladas via **OpenCV**.
3. O algoritmo de agrupamento **DBSCAN** é aplicado para segmentar as colunas.
4. Cada cor representa uma casa decimal (centenas, dezenas, unidades).
5. O valor total é calculado automaticamente e retornado em formato estruturado (ex: JSON).

---

## 🧰 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/<usuario>/api-modelo-ia.git
cd api-modelo-ia
```
### 2. Instalar dependências
```bash
pip install -r requirements.txt
```
### 3. Executar o script principal
```bash
python main.py
```
> ⚠️ Certifique-se de possuir o **Python 3.10+** instalado e as bibliotecas listadas no `requirements.txt`.

---

## 📂 Estrutura do Projeto

```
api-modelo-ia/
│
├── main.py                # Núcleo da lógica de contagem (AbacusEducationalCounterV3)
├── bolinhas.py            # Versão auxiliar de detecção por cor (RobustAbacusCounter)
├── requirements.txt        # Dependências do projeto
├── images/                # Imagens de teste e validação
└── .github/workflows/     # Pipelines de CI/CD (GitHub Actions)
```

---

## 📊 Exemplo de Saída

```json
{
  "azul": 3,
  "vermelho": 2,
  "amarelo": 5,
  "total": 325
}
```

---

## 👩‍💻 Autores

**IARA Tech**

Projeto Interdisciplinar desenvolvido por alunos do **1º e 2º ano do Instituto J&F**, com o propósito de **otimizar o registro e monitoramento de ábacos industriais**.

📍 São Paulo, Brasil
📧 [iaratech.oficial@gmail.com](mailto:iaratech.oficial@gmail.com)
🌐 [https://github.com/IARA-TECH](https://github.com/IARA-TECH)

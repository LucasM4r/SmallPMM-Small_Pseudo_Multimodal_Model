# Multimodal DETR LLM-Based Image-to-Text System

## Descrição

Este repositório apresenta um sistema inovador de geração de descrições textuais a partir de imagens, combinando o poder do DETR (Detection Transformer) para detecção de objetos com modelos de linguagem de larga escala (LLMs). O objetivo é avançar o estado da arte em tarefas multimodais, como compreensão de imagens, sumarização visual e geração automática de legendas, com aplicações em acessibilidade, indexação de imagens e sistemas inteligentes.

## Motivação

A integração de modelos de detecção de objetos e LLMs permite uma compreensão mais profunda do conteúdo visual, possibilitando descrições ricas e contextuais. Este trabalho será submetido como artigo científico, detalhando arquitetura, experimentos e resultados.

## Arquitetura

- **DETR (Detection Transformer):** Utilizado para identificar e localizar objetos em imagens.
- **LLM (Large Language Model):** Responsável por gerar descrições textuais a partir das informações extraídas pelo DETR.
- **FiftyOne:** Ferramenta para visualização, avaliação e manipulação de datasets de visão computacional.

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/multimodal_detr_llm_based_image_to_text_architecture.git
    cd multimodal_detr_llm_based_image_to_text_architecture
    ```

2. Crie e ative um ambiente virtual:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Execute o script principal para processar imagens e gerar descrições:

```bash
python src/detr_resnet_model.py
```

O sistema irá:
- Carregar um dataset de imagens (COCO 2017, por padrão)
- Detectar objetos usando DETR
- Gerar e avaliar descrições textuais
- Exibir resultados via FiftyOne

## Estrutura do Projeto

```
multimodal_detr_llm_based_image_to_text_architecture/
│
├── src/
│   └── detr_resnet_model.py
├── requirements.txt
├── README.md
└── venv/
```

## Requisitos

- Python 3.8+
- PyTorch
- Transformers
- Pillow
- FiftyOne

## Referências

- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

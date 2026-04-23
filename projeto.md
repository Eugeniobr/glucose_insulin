# 📊 Sistema de Modelagem Glicose–Insulina (Tempo Real)

## 🎯 Objetivo

Construir um sistema que:

1. Lê glicemia contínua (JSON)
2. Recebe entradas manuais (refeições + insulina) via Google Forms
3. Resolve um **problema inverso**
4. Executa o **modelo forward**
5. Compara simulação vs observado
6. Atualiza a cada **15 minutos**

---

# 🧠 Visão Geral

```text
Celular (Forms)
      ↓
Google Sheets (CSV)
      ↓
Python Pipeline
      ↓
[Inverso → Forward]
      ↓
Outputs + Gráficos
```

---

# 📁 Estrutura do Projeto

```text
glicose_model/
├── data/
│   ├── dados_glicose_final.json
│
├── src/
│   ├── data_io.py
│   ├── inputs.py
│   ├── model.py
│   ├── inverse.py
│   ├── forward.py
│   └── runner.py
│
├── outputs/
│   ├── plots/
│   ├── metrics.json
│   └── state_latest.json
│
├── config.py
├── main.py
└── README.md
```

---

# 📊 Dados de Entrada

## 1. Glicose (JSON)

Formato:

```json
{
  "timestamp": "2026-04-18 11:29:37",
  "value_in_mg_per_dl": 248.0
}
```

Frequência: ~15 min

---

## 2. Refeições (Google Forms → CSV)

Colunas esperadas:

```text
Timestamp | Horário | Carboidratos (g)
```

---

## 3. Insulina (Google Forms → CSV)

Colunas:

```text
Timestamp | Horário | Unidades | Tipo
```

---

# 🔗 Integração com Google Sheets

Publicar como CSV:

```text
Arquivo → Compartilhar → Publicar na web → CSV
```

Exemplo de leitura:

```python
df = pd.read_csv(URL)
```

---

# 🧠 Modelo Matemático

## Estados

* (S(t)): insulina subcutânea
* (I_p(t)): insulina plasmática
* (I_i(t)): insulina remota
* (G(t)): glicose
* (h_1,h_2,h_3): atraso

---

## Equações

### Subcutâneo

[
\frac{dS}{dt} = -k_a S + D(t)
]

---

### Insulina plasmática

[
\frac{dI_p}{dt}
===============

## f_1(G)

## \left(\frac{I_p}{V_p}-\frac{I_i}{V_i}\right)E

\frac{I_p}{t_p}
+
k_a S
]

---

### Insulina remota

[
\frac{dI_i}{dt}
===============

## \left(\frac{I_p}{V_p}-\frac{I_i}{V_i}\right)E

\frac{I_i}{t_i}
]

---

### Glicose

[
\frac{dG}{dt}
=============

f_4(h_3)+I_G(t)-f_2(G)-f_3(I_i)G
]

---

### Atraso

[
t_d\dot h_1=I_p-h_1
]
[
t_d\dot h_2=h_1-h_2
]
[
t_d\dot h_3=h_2-h_3
]

---

# 🍽️ Modelagem de Refeição

[
I_G(t) = \sum_k A_k (t-t_k)e^{-(t-t_k)/\tau}
]

---

# 💉 Insulina Exógena

Bolus:

```python
S(t_j^+) = S(t_j^-) + dose
```

---

# 🔄 Pipeline

## Etapas

1. Ler glicose
2. Ler refeições
3. Ler insulina
4. Converter timestamps → minutos
5. Resolver problema inverso
6. Rodar forward
7. Comparar com observado
8. Salvar outputs

---

# 🧠 Problema Inverso

Objetivo:

[
\min \sum (G_{sim}(t) - G_{obs}(t))^2
]

Parâmetros estimados:

* estados iniciais
* (k_a)
* escala da absorção
* bioavailability

---

# 🚀 Execução Contínua

## main.py

```python
while True:
    update_pipeline()
    time.sleep(900)
```

---

# 📊 Outputs

Salvar:

* gráfico glicose real vs simulada
* insulina estimada
* reservatório S
* métricas

---

# 📈 Métricas

* RMSE
* erro absoluto médio
* erro de pico
* atraso de pico

---

# 🔥 Otimizações Futuras

* janela móvel (últimas 12h)
* warm start
* previsão futura (1–2h)
* separação basal vs rápida
* dashboard

---

# ⚠️ Cuidados

* alinhar timestamps corretamente
* evitar overfitting
* manter parâmetros fisiológicos fixos inicialmente

---

# 🎯 Objetivo Final

Sistema que permita:

* inferir dinâmica insulina–glicose
* validar modelo em tempo real
* prever glicemia futura

---

# 🚀 Próximo Passo

Implementar:

```python
update_pipeline()
```

como função central que:

* orquestra todo o fluxo
* integra todos os módulos
* salva resultados

---

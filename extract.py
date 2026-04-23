import json
import os
from pylibrelinkup.pylibrelinkup import PyLibreLinkUp
from pylibrelinkup.api_url import APIUrl

EMAIL = os.getenv("LIBRELINK_EMAIL", "").strip()
PASSWORD = os.getenv("LIBRELINK_PASSWORD", "").strip()

def extrair_dados():
    try:
        if not EMAIL or not PASSWORD:
            raise RuntimeError("Defina LIBRELINK_EMAIL e LIBRELINK_PASSWORD no ambiente.")
        # 1. Inicialização com servidor LA
        client = PyLibreLinkUp(
            email=EMAIL, 
            password=PASSWORD, 
            api_url=APIUrl.LA
        )
        
        # 2. Autenticação
        print(f"Autenticando {EMAIL}...")
        client.authenticate()
        
        # 3. Obter lista de pacientes
        patients = client.get_patients()
        if not patients:
            print("Nenhum paciente encontrado. Verifique o convite no app.")
            return

        patient = patients[0]
        print(f"Paciente encontrado: {patient.first_name}")

        # 4. Extraindo os dados do gráfico
        print("Extraindo dados do gráfico...")
        graph_data = client.graph(patient)
        
        # 5. Conversão dinâmica (Pega todos os atributos disponíveis)
        json_ready_data = []
        for m in graph_data:
            # vars(m) ou m.__dict__ pega todos os campos do objeto automaticamente
            data_point = vars(m)
            # Removemos atributos internos que começam com '_' para o JSON ficar limpo
            clean_point = {k: v for k, v in data_point.items() if not k.startswith('_')}
            json_ready_data.append(clean_point)

        # Salva o resultado final
        with open('dados_glicose_final.json', 'w') as f:
            json.dump(json_ready_data, f, indent=4, default=str)
            
        print("\n" + "="*40)
        print("🚀 SUCESSO ABSOLUTO!")
        print(f"Foram extraídos {len(json_ready_data)} pontos de glicose.")
        print(f"Arquivo salvo: ~/Documents/glucose_insulin/dados_glicose_final.json")
        print("="*40)

    except Exception as e:
        print(f"Erro no experimento: {e}")

if __name__ == "__main__":
    extrair_dados()

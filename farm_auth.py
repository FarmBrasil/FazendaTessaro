# Nome do arquivo: farm_auth.py
import requests
import os
import sys

# URL de login
LOGIN_URL = "https://admin.farmcommand.com/login/"

def get_authenticated_session() -> requests.Session | None:
    # Pega usuário e senha direto do "Cofre" do GitHub (Secrets)
    USUARIO = os.environ.get("FARM_USER") 
    SENHA = os.environ.get("FARM_PASS")
    
    if not USUARIO or not SENHA:
        print("❌ ERRO: Usuário ou Senha não encontrados nas Variáveis de Ambiente.")
        return None

    print("\n--- [farm_auth] Tentando Login... ---")
    
    s = requests.Session()
    try:
        # 1. Acessa a página para pegar o token de segurança
        login_page = s.get(LOGIN_URL)
        login_page.raise_for_status() 
        
        if 'csrftoken' not in s.cookies:
            print("Erro: Token de segurança não encontrado.")
            return None
        csrftoken = s.cookies['csrftoken']

        # 2. Manda usuário e senha
        login_data = {
            'username': USUARIO,
            'password': SENHA,
            'csrfmiddlewaretoken': csrftoken
        }
        headers = {'Referer': LOGIN_URL}

        r_login = s.post(LOGIN_URL, data=login_data, headers=headers)
        r_login.raise_for_status() 

        # 3. Verifica se deu certo
        if 'login' in r_login.url:
            print("❌ Falha no login. Verifique usuário e senha.")
            return None
        
        print("✅ Login realizado com sucesso!")
        
        # Configura a sessão para as próximas chamadas
        s.headers.update({
            'X-CSRFToken': s.cookies.get('csrftoken'),
            "Referer": "https://admin.farmcommand.com/"
        })
        
        return s

    except Exception as e:
        print(f"Erro no login: {e}")
        return None

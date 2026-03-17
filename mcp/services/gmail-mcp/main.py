import os
import os.path
import json
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from fastmcp import FastMCP

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gmail-mcp")

app = FastMCP("gmail-mcp")

def get_gmail_service():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                logger.error("credentials.json not found. Please provide Gmail OAuth credentials.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

@app.tool()
def list_emails(max_results: int = 10, query: str = "") -> dict:
    """
    List recent emails from Gmail.
    """
    service = get_gmail_service()
    if not service:
        return {"status": "error", "message": "Gmail service not initialized. Check credentials.json."}
    
    try:
        results = service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()
        messages = results.get('messages', [])
        
        email_summaries = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='minimal').execute()
            # Extract snippet or other metadata if needed
            email_summaries.append({
                "id": msg['id'],
                "snippet": msg_data.get('snippet', '')
            })
            
        return {"emails": email_summaries}
    except Exception as e:
        logger.error(f"Error listing emails: {e}")
        return {"status": "error", "message": str(e)}

@app.tool()
def read_email(email_id: str) -> dict:
    """
    Read the content of a specific email by ID.
    """
    service = get_gmail_service()
    if not service:
        return {"status": "error", "message": "Gmail service not initialized."}
    
    try:
        msg = service.users().messages().get(userId='me', id=email_id).execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
        
        return {
            "id": email_id,
            "subject": subject,
            "from": sender,
            "snippet": msg.get('snippet', ''),
            "body": msg.get('snippet', '') # For simplicity, returning snippet as body
        }
    except Exception as e:
        logger.error(f"Error reading email: {e}")
        return {"status": "error", "message": str(e)}

@app.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """
    Send an email via Gmail.
    """
    service = get_gmail_service()
    if not service:
        return {"status": "error", "message": "Gmail service not initialized."}
    
    import base64
    from email.message import EmailMessage

    try:
        message = EmailMessage()
        message.set_content(body)
        message['To'] = to
        message['From'] = 'me'
        message['Subject'] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        
        send_message = service.users().messages().send(userId="me", body=create_message).execute()
        return {"status": "success", "message_id": send_message['id']}
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    app.run()

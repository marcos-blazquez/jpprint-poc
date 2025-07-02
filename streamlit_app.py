import streamlit as st
import boto3
import json
import os
import uuid
from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PixPod - PoC",
    page_icon="🤖",
    layout="wide"
)

def initialize_aws_client():
    """Initialize AWS Bedrock client with multiple credential sources"""
    try:
        # Method 1: Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'AWS_ACCESS_KEY_ID' in st.secrets:
            client = boto3.client(
                'bedrock-agent-runtime',
                region_name='us-east-1',
                aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY'],
                aws_session_token=st.secrets.get('AWS_SESSION_TOKEN')  # Optional
            )
            st.success("✅ AWS credentials loaded from Streamlit secrets")
            return client
            
    except Exception as e:
        st.warning(f"Streamlit secrets method failed: {e}")
    
    try:
        # Method 2: Try environment variables
        if os.getenv('AWS_ACCESS_KEY_ID'):
            client = boto3.client(
                'bedrock-agent-runtime',
                region_name='us-east-1',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                aws_session_token=os.getenv('AWS_SESSION_TOKEN')  # Optional
            )
            st.success("✅ AWS credentials loaded from environment variables")
            return client
            
    except Exception as e:
        st.warning(f"Environment variables method failed: {e}")
    
    try:
        # Method 3: Try default AWS credentials (IAM role, ~/.aws/credentials, etc.)
        client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
        # Test the connection
        client.list_agents(maxResults=1)
        st.success("✅ AWS credentials loaded from default sources")
        return client
        
    except Exception as e:
        st.error(f"Default credentials method failed: {e}")
        return None

def get_agent_config():
    """Get agent configuration from secrets or environment"""
    agent_id = None
    agent_alias_id = None
    
    # Try Streamlit secrets
    try:
        agent_id = st.secrets['AGENT_ID']
        agent_alias_id = st.secrets['AGENT_ALIAS_ID']
    except:
        # Try environment variables
        agent_id = os.getenv('AGENT_ID')
        agent_alias_id = os.getenv('AGENT_ALIAS_ID')
    
    return agent_id, agent_alias_id

def process_response(resp):
    """Process the Bedrock agent response"""
    event_stream = resp['completion']
    try:
        for event in event_stream:
            if 'chunk' in event:
                data = event['chunk']['bytes']                              
                agent_answer = data.decode('utf8')
                return agent_answer
            else:
                raise Exception("unexpected event.", event)
    except Exception as e:
        raise Exception("unexpected event.", e)

def generate_response(client, prompt, agent_id, agent_alias_id, session_id):
    """Generate response from Bedrock agent with error handling"""
    try:
        with st.spinner("🤖 Thinking..."):
            response = client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
            )
            
            # Use the original parsing function
            response_text = process_response(response)
            return response_text if response_text else "No response received"
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            return "❌ Access denied. Check your AWS permissions for Bedrock."
        elif error_code == 'ResourceNotFoundException':
            return "❌ Agent not found. Check your Agent ID and Alias ID."
        else:
            return f"❌ AWS Error: {e.response['Error']['Message']}"
            
    except NoCredentialsError:
        return "❌ AWS credentials not found. Please configure credentials."
        
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'client_initialized' not in st.session_state:
    st.session_state.client_initialized = False

if 'client' not in st.session_state:
    st.session_state.client = None

# App header
st.title("🤖 PixPod - Prueba de concepto")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Session info
    st.subheader("📋 Session Info")
    st.code(f"Session ID: {st.session_state.session_id[:8]}...")
    
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # AWS Configuration
    st.subheader("☁️ AWS Status")
    
    if not st.session_state.client_initialized:
        if st.button("🔌 Initialize AWS Client"):
            st.session_state.client = initialize_aws_client()
            st.session_state.client_initialized = True
            st.rerun()
    else:
        if st.session_state.client:
            st.success("✅ AWS Client Ready")
        else:
            st.error("❌ AWS Client Failed")
            if st.button("🔄 Retry Connection"):
                st.session_state.client = initialize_aws_client()
                st.rerun()
    
    # Agent Configuration
    st.subheader("🤖 Agent Config")
    agent_id, agent_alias_id = get_agent_config()
    
    if agent_id and agent_alias_id:
        st.success("✅ Agent config loaded")
        st.code(f"Agent: {agent_id[:8]}...")
        st.code(f"Alias: {agent_alias_id}")
    else:
        st.error("❌ Missing agent configuration")
        st.info("Add AGENT_ID and AGENT_ALIAS_ID to secrets")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")

# Check if everything is configured
if not st.session_state.client_initialized:
    st.warning("⚠️ Please initialize AWS client in the sidebar first.")
    st.stop()

if not st.session_state.client:
    st.error("❌ AWS client not available. Check credentials and try again.")
    
    # Configuration help
    with st.expander("🔧 How to configure AWS credentials"):
        st.markdown("""
        **Method 1: Streamlit Secrets (Recommended for Codespaces)**
        
        Create `.streamlit/secrets.toml`:
        ```toml
        AWS_ACCESS_KEY_ID = "your-access-key"
        AWS_SECRET_ACCESS_KEY = "your-secret-key"
        AWS_SESSION_TOKEN = "your-session-token"  # Optional
        AGENT_ID = "your-agent-id"
        AGENT_ALIAS_ID = "your-agent-alias-id"
        ```
        
        **Method 2: Environment Variables**
        ```bash
        export AWS_ACCESS_KEY_ID="your-access-key"
        export AWS_SECRET_ACCESS_KEY="your-secret-key"
        export AGENT_ID="your-agent-id"
        export AGENT_ALIAS_ID="your-agent-alias-id"
        ```
        
        **Method 3: AWS CLI Configuration**
        ```bash
        aws configure
        ```
        """)
    st.stop()

agent_id, agent_alias_id = get_agent_config()
if not (agent_id and agent_alias_id):
    st.error("❌ Agent configuration missing. Check AGENT_ID and AGENT_ALIAS_ID in secrets.")
    st.stop()

# Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        # Use text to preserve exact formatting
        st.chat_message("assistant").text(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Generate response
    response = generate_response(
        st.session_state.client, 
        prompt, 
        agent_id, 
        agent_alias_id, 
        st.session_state.session_id
    )
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Use text to preserve exact formatting
    st.chat_message("assistant").text(response)

# Footer with usage stats
with col2:
    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    if st.session_state.messages:
        st.subheader("📥 Export Chat")
        chat_export = {
            "session_id": st.session_state.session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        
        st.download_button(
            label="💾 Download Chat",
            data=json.dumps(chat_export, indent=2),
            file_name=f"chat_{st.session_state.session_id[:8]}.json",
            mime="application/json"
        )
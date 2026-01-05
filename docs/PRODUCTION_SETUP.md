# Production Setup Guide
## Linear Regression Guide with AI Interpretation

This guide covers setting up the application for production with all features including the AI-powered model interpretation.

## Prerequisites

### System Requirements
- Python 3.9+
- pip package manager
- Streamlit-compatible hosting (Streamlit Cloud, Railway, Heroku, etc.)

### Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit`: Web framework
- `numpy`, `pandas`: Data processing
- `plotly`: Interactive visualizations
- `statsmodels`: Statistical modeling
- `scipy`: Scientific computing
- `requests`: External API calls
- `openai`: Perplexity API integration

## Configuration

### 1. Perplexity API Setup

#### Get API Key
1. Visit [Perplexity API Settings](https://www.perplexity.ai/settings/api)
2. Create account or sign in
3. Generate API key

#### Configure API Key

**Option A: Environment Variable (Recommended for production)**
```bash
export PERPLEXITY_API_KEY="your-perplexity-api-key-here"
```

**Option B: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
PERPLEXITY_API_KEY = "your-perplexity-api-key-here"
```

### 2. Application Configuration

#### Environment Variables
```bash
# Set Python path for proper imports
export PYTHONPATH="${PYTHONPATH}:/app/src"

# Streamlit configuration
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

#### Streamlit Config (.streamlit/config.toml)
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
base = "light"
```

## Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://share.streamlit.io/)
3. Set environment variable: `PERPLEXITY_API_KEY`
4. Deploy

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "run.py", "--server.headless", "true"]
```

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PERPLEXITY_API_KEY="your-key-here"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run application
streamlit run run.py
```

## Testing

### Pre-Deployment Tests
```bash
# Test API configuration
python -c "from src.perplexity_api import is_api_configured; print('API configured:', is_api_configured())"

# Test imports
python -c "from src.app import main; print('App imports successful')"

# Run basic functionality tests
python -m pytest tests/ -v --tb=short
```

### Production Health Checks
- ✅ API key configured correctly
- ✅ All imports working
- ✅ R output rendering functional
- ✅ AI interpretation button appears
- ✅ Basic statistical computations work

## Security Considerations

### API Key Security
- ✅ Never commit API keys to version control
- ✅ Use environment variables in production
- ✅ Rotate keys regularly
- ✅ Monitor API usage for anomalies

### Data Privacy
- ✅ Only statistical summaries sent to API (no raw data)
- ✅ Transparent prompt display for user awareness
- ✅ No user data stored beyond session
- ✅ No personal information collected

## Performance Optimization

### API Cost Management
- Each AI interpretation costs API credits
- Consider implementing caching for repeated requests
- Monitor usage through Perplexity dashboard

### Application Performance
- Streamlit sessions are stateless by default
- Large datasets may impact performance
- Consider implementing data caching for repeated computations

## Monitoring & Maintenance

### Logs
Application logs important events:
- API configuration status
- AI interpretation requests/results
- Error conditions

### Updates
- Regularly update dependencies for security
- Monitor Perplexity API changes
- Test new Streamlit versions before upgrading

## Troubleshooting

### Common Issues

**"API nicht konfiguriert"**
- Check API key is set correctly
- Verify environment variable name
- Restart application after key changes

**Import Errors**
- Ensure PYTHONPATH includes `/app/src`
- Check all dependencies are installed
- Verify file permissions

**AI Interpretation Fails**
- Check API key validity and credits
- Verify network connectivity
- Check Perplexity service status

## Support

For issues:
1. Check application logs
2. Verify configuration matches this guide
3. Test API key independently
4. Contact development team

---

**Status**: ✅ Production-ready with AI interpretation
**API Key**: ✅ Configured
**Documentation**: ✅ Complete
**Security**: ✅ Verified
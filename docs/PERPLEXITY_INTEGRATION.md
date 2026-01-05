# Perplexity API Integration

## Overview

This feature adds AI-powered interpretation of regression model results using the Perplexity API. The interpretation appears next to the R output display and provides insights about model quality, coefficient interpretation, residuals, and practical recommendations.

## Features

1. **AI Interpretation Button**: Generates a comprehensive interpretation of the current regression model
2. **Prompt Transparency**: View and download the exact data sent to the AI
3. **Easy Data Export**: Download the prompt as a text file or copy it to clipboard

## Setup

### Getting a Perplexity API Key

1. Visit [Perplexity API Settings](https://www.perplexity.ai/settings/api)
2. Create an account or sign in
3. Generate an API key

### Configuration

You can configure the API key in two ways:

#### Option 1: Environment Variable

```bash
export PERPLEXITY_API_KEY="your-api-key-here"
```

Then start the application:

```bash
streamlit run run.py
```

#### Option 2: Streamlit Secrets

Create or edit `.streamlit/secrets.toml`:

```toml
PERPLEXITY_API_KEY = "your-api-key-here"
```

## Usage

1. Select a dataset and configure parameters in the sidebar
2. The R output will be displayed automatically
3. In the right column, find the **🤖 AI-Interpretation** section
4. Click **"🔍 Interpretation generieren"** to get an AI interpretation
5. View the interpretation below the button
6. Optionally, expand **"📋 An AI gesendete Daten anzeigen"** to:
   - View the exact prompt sent to the API
   - Download the prompt as a text file
   - Copy the prompt text manually

## What Gets Interpreted

The AI receives structured information about:

- Model type and basic statistics (n, R², F-statistic, etc.)
- All coefficients with their estimates, standard errors, t-values, and p-values
- Residual summary statistics (min, Q1, median, Q3, max)
- Model fit quality metrics

## Interpretation Structure

The AI provides interpretation covering:

1. **Model Quality**: How good is the model? What do R² and F-statistic tell us?
2. **Coefficient Interpretation**: What do the coefficients mean in practice? Which predictors are significant?
3. **Residuals**: What do the residuals tell us about model fit?
4. **Practical Significance**: What are the key insights for practice?
5. **Recommendations**: Are there hints for improvement or precautions?

## API Costs

The Perplexity API is a paid service. Each interpretation request consumes API credits. The amount depends on:
- The model used (default: `llama-3.1-sonar-large-128k-online`)
- The length of the prompt (varies with model complexity)
- The length of the response

Check [Perplexity's pricing](https://www.perplexity.ai/hub/pricing) for current rates.

## Privacy & Security

- The API key is never displayed in the UI
- Only statistical summaries are sent to the API (no raw data)
- The prompt is displayed transparently so users can see exactly what data is shared
- No data is stored by this application beyond the session

## Troubleshooting

### "API nicht konfiguriert" Warning

**Problem**: The API key is not set or not found.

**Solution**: 
1. Verify the API key is correctly set in environment variables or secrets.toml
2. Restart the Streamlit application after setting the key
3. Check for typos in the variable name (must be exactly `PERPLEXITY_API_KEY`)

### API Request Errors

**Problem**: Error message when generating interpretation.

**Possible causes**:
- Invalid API key
- Insufficient API credits
- Network connectivity issues
- API service temporarily unavailable

**Solution**:
- Check your API key is valid and has credits
- Try again after a moment
- Check the application logs for detailed error messages

## Technical Details

### Modules

- `src/perplexity_api.py`: Core API integration logic
- `src/r_output.py`: UI components for displaying interpretations

### Key Functions

- `extract_model_statistics()`: Extracts relevant statistics from statsmodels regression model
- `create_interpretation_prompt()`: Formats a structured prompt for the AI
- `get_interpretation_from_perplexity()`: Calls the Perplexity API
- `interpret_model()`: High-level function combining extraction and API call

### Dependencies

- `openai>=1.0.0`: For API communication (Perplexity uses OpenAI-compatible API)
- `streamlit`: For UI components
- `numpy`, `statsmodels`: For statistical computations

## Future Enhancements

Potential improvements for future versions:

- [ ] Support for multiple AI models (GPT-4, Claude, etc.)
- [ ] Caching interpretations to reduce API costs
- [ ] Comparative interpretations (before/after model changes)
- [ ] Export interpretations as PDF reports
- [ ] Custom prompt templates
- [ ] Language selection (currently German only)

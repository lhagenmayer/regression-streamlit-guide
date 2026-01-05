# Deployment Guide - Streamlit Cloud

This guide provides step-by-step instructions for deploying the Linear Regression Guide to Streamlit Cloud.

## Prerequisites

- GitHub account with access to this repository
- Streamlit Cloud account (free tier available at [share.streamlit.io](https://share.streamlit.io))

## Repository Preparation

The repository is already configured for Streamlit Cloud deployment:

✅ **Dependencies**: All required packages are listed in `requirements.txt`
✅ **Configuration**: Streamlit config is in `.streamlit/config.toml`
✅ **Data**: All datasets are simulated/offline (no API keys needed)
✅ **Python Version**: Compatible with Python 3.9-3.12

## Deployment Steps

### 1. Create Streamlit Cloud Account

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up" and authenticate with your GitHub account
3. Authorize Streamlit to access your GitHub repositories

### 2. Deploy the Application

1. **Navigate to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" button

2. **Configure Deployment Settings**
   - **Repository**: `yourusername/linear-regression-guide` (your forked repository)
   - **Branch**: `main` (or your preferred branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (e.g., `linear-regression-guide`)

3. **Advanced Settings** (Optional)
   - **Python version**: 3.9, 3.10, 3.11, or 3.12 (recommended: 3.11)
   - **Secrets**: Not required for this app
   
4. **Deploy**
   - Click "Deploy!" button
   - Wait for the app to build (typically 2-5 minutes)
   - The app will be available at your chosen URL

### 3. Post-Deployment Verification

Once deployed, verify the following:

#### Functionality Checklist
- [ ] App loads without errors
- [ ] All tabs are accessible and functional
- [ ] Simulated dataset generates correctly
- [ ] Real city dataset loads properly
- [ ] Interactive plots render correctly
- [ ] Sidebar controls work (sliders, selectors, etc.)
- [ ] Regression calculations display properly
- [ ] R-style output formats display correctly
- [ ] ANOVA tables render properly
- [ ] 3D visualizations work in all tabs

#### Performance Checklist
- [ ] Initial load time < 10 seconds
- [ ] Tab switching is smooth
- [ ] Parameter changes update quickly
- [ ] No memory warnings in logs
- [ ] Plots render within 2-3 seconds

#### Mobile Responsiveness
- [ ] Layout adapts to mobile screens
- [ ] Sidebar is accessible on mobile
- [ ] Plots are readable on small screens
- [ ] Touch interactions work properly

### 4. Custom Domain Setup (Optional)

To use a custom domain:

1. In Streamlit Cloud, go to your app settings
2. Navigate to "Custom domain" section
3. Follow the DNS configuration instructions
4. Add CNAME record pointing to Streamlit Cloud
5. Wait for DNS propagation (5-30 minutes)

### 5. Sharing and Access Control

**Public Access** (Default):
- App is publicly accessible to anyone with the URL
- No authentication required
- Suitable for educational content

**Password Protection** (Optional):
- Upgrade to Streamlit Cloud Pro plan
- Enable password protection in app settings
- Share password with intended users

## Monitoring and Maintenance

### Application Logs

Access logs in Streamlit Cloud:
1. Go to your app dashboard
2. Click "Manage app"
3. View logs in the "Logs" tab

### Common Issues and Solutions

#### Issue: App fails to start
**Solution**: Check logs for import errors or missing dependencies

#### Issue: Slow performance
**Solutions**:
- Verify caching decorators are in place (`@st.cache_data`)
- Check data generation functions are optimized
- Review Streamlit Cloud resource limits

#### Issue: Plots not rendering
**Solution**: Verify plotly version matches requirements.txt

#### Issue: Out of memory errors
**Solutions**:
- Reduce default dataset sizes in config.py
- Optimize data caching strategies
- Consider upgrading Streamlit Cloud tier

### Automatic Updates

Streamlit Cloud automatically redeploys when you push to the configured branch:

1. Make changes locally
2. Commit and push to GitHub
3. Streamlit Cloud detects changes
4. Automatic rebuild and redeploy (2-5 minutes)

## Configuration Files

### `.streamlit/config.toml`
Contains Streamlit-specific configuration:
- Theme colors (primaryColor, backgroundColor, etc.)
- Server settings (CORS, upload limits, etc.)
- Browser defaults
- Runner behavior

### `requirements.txt`
Lists all Python dependencies:
```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.14.0
statsmodels>=0.14.0
scipy>=1.11.0
```

### No System Dependencies Required
This app uses only pure Python packages - no `packages.txt` needed.

## Environment Variables and Secrets

This application does not require any environment variables or secrets:
- ✅ No API keys needed
- ✅ No database connections
- ✅ No external service credentials
- ✅ All data is simulated or offline

## Performance Optimization

The app includes several performance optimizations for cloud deployment:

1. **Data Caching**: `@st.cache_data` decorators on data generation functions
2. **Session State**: Model results cached in session state
3. **Lazy Loading**: Tabs load content only when accessed
4. **Efficient Plotting**: Optimized plotly configurations
5. **Smart Recalculation**: Updates only when parameters change

## Troubleshooting

### Build Failures

If deployment fails, check:
1. Python version compatibility (3.9-3.12)
2. Dependencies in requirements.txt are valid
3. No syntax errors in Python files
4. No missing imports or files

### Runtime Errors

If app crashes after deployment:
1. Check Streamlit Cloud logs
2. Verify all imports work correctly
3. Test locally with same Python version
4. Check for missing configuration files

### Performance Issues

If app is slow on Streamlit Cloud:
1. Verify caching is working (check logs)
2. Reduce default dataset sizes
3. Optimize plot generation
4. Consider data precomputation

## Support and Resources

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Streamlit Cloud Docs**: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- **GitHub Issues**: Report issues in this repository

## Rollback Procedure

If a deployment causes issues:

1. **Immediate Fix**:
   - In Streamlit Cloud, click "Reboot app"
   - Or temporarily disable the app

2. **Version Rollback**:
   - Revert the problematic commit in GitHub
   - Push the revert to trigger redeployment
   - Or change the deployment branch in Streamlit Cloud settings

3. **Emergency Shutdown**:
   - Delete the app in Streamlit Cloud dashboard
   - Redeploy from a known good commit

## Security Considerations

✅ **No sensitive data**: App uses only simulated data
✅ **No authentication needed**: Public educational tool
✅ **No user data collection**: No forms or user inputs stored
✅ **XSRF protection enabled**: Default Streamlit security
✅ **No external APIs**: No API keys or credentials exposed

## Cost Considerations

**Free Tier Limits** (Streamlit Cloud Community):
- 1 private app or unlimited public apps
- 1 GB RAM per app
- Shared CPU resources
- 1 GB storage

This app runs well within free tier limits:
- Memory usage: ~200-400 MB typical
- CPU: Low (only during data generation)
- Storage: <10 MB (no persistent data)

## License and Attribution

When deploying, ensure:
- MIT License is preserved
- Attribution is maintained in the app
- No license violations in dependencies

## Next Steps After Deployment

1. **Test thoroughly**: Go through all features
2. **Monitor performance**: Check logs regularly for first week
3. **Share the URL**: Add to README and documentation
4. **Collect feedback**: Note any user-reported issues
5. **Iterate**: Make improvements based on usage patterns

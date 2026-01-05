# Accessibility Guide

This document describes the accessibility features implemented in the Linear Regression Guide application.

## Overview

The application has been enhanced with accessibility features to ensure it's usable by people with disabilities, including:

- **Visual accessibility**: High contrast colors, focus indicators, responsive design
- **Keyboard navigation**: Full keyboard support with visible focus states
- **Screen reader support**: ARIA labels, semantic HTML, descriptive text
- **Cognitive accessibility**: Clear structure, consistent patterns, loading indicators

## Accessibility Features

### 1. Visual Accessibility

#### High Contrast Colors

All interactive elements use high contrast color combinations that meet WCAG 2.1 AA standards:

- Primary color: `#1f77b4` (blue) with white text
- Text color: `#0d3d66` (dark blue) on white backgrounds
- Contrast ratio: > 4.5:1 for normal text, > 3:1 for large text

#### Focus Indicators

All interactive elements have clear focus indicators for keyboard navigation:

- **Outline**: 3px solid blue outline with 2px offset
- **Box shadow**: Additional shadow for emphasis
- **Visible on**: Buttons, inputs, selects, textareas, links

```css
button:focus {
    outline: 3px solid #1f77b4;
    outline-offset: 2px;
    box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.3);
}
```

#### Responsive Design

The application adapts to different screen sizes:

- **Mobile-friendly**: Font sizes and layouts adjust for small screens
- **Tablet-optimized**: Medium breakpoints for tablets
- **Desktop**: Full-featured layout for large screens

```css
@media (max-width: 768px) {
    .main-header { font-size: 2rem; }
    .section-header { font-size: 1.5rem; }
}
```

### 2. Keyboard Navigation

#### Tab Order

All interactive elements can be reached using the Tab key:

1. Navigation tabs
2. Sidebar controls (sliders, selectboxes, buttons)
3. Main content (plots, expandable sections)
4. Footer links

#### Keyboard Shortcuts

Standard browser keyboard shortcuts work:

- **Tab**: Move to next interactive element
- **Shift+Tab**: Move to previous interactive element
- **Enter/Space**: Activate buttons and links
- **Arrow keys**: Navigate within sliders and dropdowns
- **Esc**: Close modals and dropdowns

### 3. Screen Reader Support

#### ARIA Labels

The application uses ARIA (Accessible Rich Internet Applications) attributes to provide context for screen readers:

```python
from accessibility import add_aria_label

# Add ARIA label to an element
aria_attrs = add_aria_label("slider", "Number of observations", 
                            "Controls the sample size for the dataset")
```

#### Screen Reader Text

Hidden text provides additional context for screen readers:

```python
from accessibility import create_screen_reader_text

# Add screen reader only text
create_screen_reader_text(
    "This scatter plot shows the relationship between X and Y variables",
    element_id="plot-description"
)
```

#### Plot Descriptions

Charts and visualizations include descriptive text for screen readers:

```python
from accessibility import add_plot_accessibility_description

add_plot_accessibility_description(
    plot_type="Scatter",
    title="Regression Analysis",
    x_label="Independent Variable",
    y_label="Dependent Variable",
    key_insights="Strong positive correlation with R² = 0.85"
)
```

### 4. Cognitive Accessibility

#### Clear Structure

- **Consistent layout**: Same structure across all tabs
- **Clear headings**: Hierarchical heading structure (h1, h2, h3)
- **Visual grouping**: Related controls grouped in expandable sections

#### Loading Indicators

Loading states inform users when operations are in progress:

```python
from accessibility import show_loading_indicator

with show_loading_indicator("Generating dataset..."):
    data = generate_large_dataset()
```

#### Help Text

All controls include descriptive help text:

```python
from accessibility import create_accessible_slider

value = create_accessible_slider(
    label="Sample Size",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    key="n_samples",
    help_text="Number of observations in the dataset. Larger samples provide more reliable results."
)
```

## Using Accessibility Features

### Basic Usage

Import the accessibility module in your Streamlit app:

```python
from accessibility import inject_accessibility_styles

# At the start of your app
inject_accessibility_styles()
```

### Creating Accessible Controls

#### Accessible Buttons

```python
from accessibility import create_accessible_button

if create_accessible_button(
    label="Generate Data",
    key="generate_btn",
    help_text="Click to generate a new random dataset"
):
    # Button was clicked
    generate_data()
```

#### Accessible Sliders

```python
from accessibility import create_accessible_slider

n_samples = create_accessible_slider(
    label="Sample Size",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    key="n_slider",
    help_text="Number of data points to generate"
)
```

#### Accessible Selectboxes

```python
from accessibility import create_accessible_selectbox

dataset = create_accessible_selectbox(
    label="Dataset",
    options=["Cities", "Houses", "Electronics"],
    index=0,
    key="dataset_select",
    help_text="Choose the type of dataset to analyze"
)
```

### Announcing Messages

Announce important messages to screen readers:

```python
from accessibility import announce_to_screen_reader

# After completing an operation
announce_to_screen_reader("Dataset generated successfully with 100 observations")
```

## Testing Accessibility

### Manual Testing

#### Keyboard Navigation

1. **Tab through the app**: Ensure all interactive elements can be reached
2. **Check focus indicators**: Verify visible focus states on all elements
3. **Test keyboard shortcuts**: Ensure Enter, Space, Arrow keys work as expected

#### Screen Reader Testing

Use screen readers to test:

- **Windows**: NVDA (free), JAWS (commercial)
- **macOS**: VoiceOver (built-in)
- **Linux**: Orca (free)

#### Color Contrast

Use tools to verify color contrast:

- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Chrome DevTools Accessibility Panel](https://developer.chrome.com/docs/devtools/accessibility/)
- [WAVE Browser Extension](https://wave.webaim.org/extension/)

### Automated Testing

#### Lighthouse Audit

Run Lighthouse in Chrome DevTools:

1. Open Chrome DevTools (F12)
2. Go to "Lighthouse" tab
3. Select "Accessibility" category
4. Click "Generate report"

Target: **Accessibility score ≥ 90**

#### axe DevTools

Use the axe browser extension:

1. Install [axe DevTools Extension](https://www.deque.com/axe/devtools/)
2. Open extension in browser
3. Run full page scan
4. Fix any issues found

## WCAG 2.1 Compliance

The application aims for **WCAG 2.1 Level AA** compliance:

### Level A (Basic)

- ✅ Text alternatives for non-text content
- ✅ Keyboard accessible
- ✅ Sufficient color contrast
- ✅ Consistent navigation

### Level AA (Standard)

- ✅ Color not sole means of conveying information
- ✅ 4.5:1 contrast ratio for normal text
- ✅ 3:1 contrast ratio for large text
- ✅ Visible focus indicators
- ✅ Multiple ways to navigate
- ✅ Consistent identification of components

## Best Practices

### Do's

✅ **Use semantic HTML**: Proper heading hierarchy, landmarks, etc.
✅ **Provide text alternatives**: Alt text, ARIA labels, descriptions
✅ **Ensure keyboard navigation**: Test with keyboard only
✅ **Maintain focus indicators**: Make focus states clearly visible
✅ **Use sufficient contrast**: Meet WCAG contrast requirements
✅ **Provide clear instructions**: Help text, labels, error messages
✅ **Test with real users**: Include people with disabilities in testing

### Don'ts

❌ **Don't rely on color alone**: Use text labels, patterns, or icons too
❌ **Don't hide focus indicators**: Users need to see where they are
❌ **Don't use vague labels**: "Click here" is not descriptive
❌ **Don't auto-play content**: Let users control audio/video
❌ **Don't use tiny touch targets**: Minimum 44x44 pixels
❌ **Don't create keyboard traps**: Users should be able to navigate away
❌ **Don't forget error messages**: Provide clear, actionable feedback

## Common Issues and Solutions

### Issue: Plots Not Accessible to Screen Readers

**Solution**: Add descriptive text using `add_plot_accessibility_description`:

```python
add_plot_accessibility_description(
    plot_type="3D Surface",
    title="Regression Plane",
    x_label="Price (CHF)",
    y_label="Advertising (CHF 1000)",
    key_insights="The regression plane shows a positive relationship with both predictors"
)
```

### Issue: Poor Color Contrast

**Solution**: Use high contrast colors from the accessibility module:

```python
# Use predefined high-contrast colors
primary_color = "#1f77b4"  # Blue
text_color = "#0d3d66"      # Dark blue
```

### Issue: Keyboard Trap

**Solution**: Ensure all modal dialogs and expandable sections can be closed with Esc key.

### Issue: Missing Form Labels

**Solution**: Use `create_accessible_*` functions which include proper labels:

```python
# Instead of plain st.slider()
value = create_accessible_slider(
    label="Clear descriptive label",
    help_text="Explanation of what this control does"
)
```

## Resources

### Guidelines and Standards

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)

### Testing Tools

- [WAVE Web Accessibility Evaluation Tool](https://wave.webaim.org/)
- [axe DevTools](https://www.deque.com/axe/devtools/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)

### Screen Readers

- [NVDA (Windows)](https://www.nvaccess.org/)
- [JAWS (Windows)](https://www.freedomscientific.com/products/software/jaws/)
- [VoiceOver (macOS/iOS)](https://www.apple.com/accessibility/voiceover/)
- [Orca (Linux)](https://help.gnome.org/users/orca/stable/)

## Future Enhancements

Potential accessibility improvements:

- [ ] Dark mode with high contrast
- [ ] Font size adjustment control
- [ ] Audio descriptions for complex visualizations
- [ ] Alternative text representations of charts (data tables)
- [ ] Skip navigation links
- [ ] Language selection for multilingual support
- [ ] Customizable color themes for color blindness
- [ ] Captions for any video content

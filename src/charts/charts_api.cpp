#include "charts_api.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace bare_nn::charts {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::string esc(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '\"') out += "&quot;";
        else out += c;
    }
    return out;
}

void start_svg(std::ostringstream& ss, const ChartConfig& cfg) {
    ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << cfg.width
       << "\" height=\"" << cfg.height << "\" viewBox=\"0 0 " << cfg.width
       << " " << cfg.height << "\">\n";
    ss << "<defs>\n";
    ss << "<linearGradient id=\"bgFade\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\n";
    ss << "<stop offset=\"0%\" stop-color=\"" << cfg.theme.background << "\"/>\n";
    ss << "<stop offset=\"100%\" stop-color=\"#dfe7ea\"/>\n";
    ss << "</linearGradient>\n";
    ss << "</defs>\n";
    ss << "<rect width=\"100%\" height=\"100%\" fill=\"url(#bgFade)\"/>\n";
    ss << "<rect x=\"16\" y=\"16\" width=\"" << (cfg.width - 32)
       << "\" height=\"" << (cfg.height - 32)
       << "\" rx=\"22\" fill=\"" << cfg.theme.panel
       << "\" stroke=\"#d8e1e5\" stroke-width=\"2\"/>\n";
    if (!cfg.title.empty()) {
        ss << "<text x=\"" << (cfg.padding) << "\" y=\"" << (cfg.padding - 20)
           << "\" fill=\"" << cfg.theme.ink
           << "\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-size=\"26\" font-weight=\"700\">"
           << esc(cfg.title) << "</text>\n";
    }
}

void end_svg(std::ostringstream& ss) {
    ss << "</svg>\n";
}

bool write_svg(const std::string& path, const std::ostringstream& ss) {
    std::ofstream out(path, std::ios::binary);
    if (!out.good()) {
        return false;
    }
    out << ss.str();
    return out.good();
}

std::string color_at(const std::vector<std::string>& palette, int i) {
    if (palette.empty()) {
        return "#4f6f7b";
    }
    return palette[static_cast<size_t>(i) % palette.size()];
}

void draw_grid(std::ostringstream& ss, const ChartConfig& cfg, int x0, int y0, int w, int h, int rows) {
    for (int i = 0; i <= rows; ++i) {
        int y = y0 + (h * i) / rows;
        ss << "<line x1=\"" << x0 << "\" y1=\"" << y << "\" x2=\"" << (x0 + w)
           << "\" y2=\"" << y << "\" stroke=\"" << cfg.theme.grid
           << "\" stroke-width=\"1\" opacity=\"0.7\"/>\n";
    }
}

}  // namespace

bool create_bar_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const ChartConfig& config) {
    if (labels.empty() || labels.size() != values.size()) {
        return false;
    }

    std::ostringstream ss;
    start_svg(ss, config);

    const int x0 = config.padding;
    const int y0 = config.padding;
    const int w = config.width - config.padding * 2;
    const int h = config.height - config.padding * 2;
    draw_grid(ss, config, x0, y0, w, h, 5);

    const double max_v = std::max(1.0, *std::max_element(values.begin(), values.end()));
    const int n = static_cast<int>(values.size());
    const double slot = static_cast<double>(w) / n;
    const double bar_w = slot * 0.62;

    for (int i = 0; i < n; ++i) {
        const double ratio = values[static_cast<size_t>(i)] / max_v;
        const double bh = ratio * (h - 30);
        const double x = x0 + i * slot + (slot - bar_w) * 0.5;
        const double y = y0 + h - bh;

        ss << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << bar_w
           << "\" height=\"" << bh << "\" rx=\"8\" fill=\"" << color_at(config.theme.palette, i)
           << "\"/>\n";

        ss << "<text x=\"" << (x + bar_w * 0.5) << "\" y=\"" << (y0 + h + 24)
           << "\" fill=\"" << config.theme.ink
           << "\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-size=\"13\" text-anchor=\"middle\">"
           << esc(labels[static_cast<size_t>(i)]) << "</text>\n";
    }

    end_svg(ss);
    return write_svg(output_svg, ss);
}

bool create_pie_chart(
    const std::string& output_svg,
    const std::vector<std::string>& labels,
    const std::vector<double>& values,
    const ChartConfig& config) {
    if (labels.empty() || labels.size() != values.size()) {
        return false;
    }

    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    if (total <= 0.0) {
        return false;
    }

    std::ostringstream ss;
    start_svg(ss, config);

    const double cx = config.width * 0.38;
    const double cy = config.height * 0.52;
    const double r = std::min(config.width, config.height) * 0.26;

    double angle = -kPi * 0.5;
    for (size_t i = 0; i < values.size(); ++i) {
        const double frac = values[i] / total;
        const double next = angle + frac * 2.0 * kPi;

        const double x1 = cx + std::cos(angle) * r;
        const double y1 = cy + std::sin(angle) * r;
        const double x2 = cx + std::cos(next) * r;
        const double y2 = cy + std::sin(next) * r;
        const int large_arc = (next - angle) > kPi ? 1 : 0;

        ss << "<path d=\"M " << cx << " " << cy << " L " << x1 << " " << y1
           << " A " << r << " " << r << " 0 " << large_arc << " 1 " << x2 << " " << y2
           << " Z\" fill=\"" << color_at(config.theme.palette, static_cast<int>(i)) << "\"/>\n";

        angle = next;
    }

    ss << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << (r * 0.48)
       << "\" fill=\"" << config.theme.panel << "\"/>\n";

    const double legend_x = config.width * 0.66;
    double legend_y = config.padding + 36;
    for (size_t i = 0; i < labels.size(); ++i) {
        const double pct = values[i] * 100.0 / total;
        ss << "<rect x=\"" << legend_x << "\" y=\"" << (legend_y - 12)
           << "\" width=\"15\" height=\"15\" rx=\"3\" fill=\"" << color_at(config.theme.palette, static_cast<int>(i))
           << "\"/>\n";
        ss << "<text x=\"" << (legend_x + 24) << "\" y=\"" << legend_y
           << "\" fill=\"" << config.theme.ink
           << "\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-size=\"14\">"
           << esc(labels[i]) << " - " << std::fixed << std::setprecision(1) << pct << "%</text>\n";
        legend_y += 28;
    }

    end_svg(ss);
    return write_svg(output_svg, ss);
}

bool create_line_chart(
    const std::string& output_svg,
    const std::vector<std::string>& x_labels,
    const std::vector<double>& values,
    const ChartConfig& config) {
    if (x_labels.empty() || x_labels.size() != values.size()) {
        return false;
    }

    std::ostringstream ss;
    start_svg(ss, config);

    const int x0 = config.padding;
    const int y0 = config.padding;
    const int w = config.width - config.padding * 2;
    const int h = config.height - config.padding * 2;
    draw_grid(ss, config, x0, y0, w, h, 6);

    const double min_v = *std::min_element(values.begin(), values.end());
    const double max_v = *std::max_element(values.begin(), values.end());
    const double span = std::max(1e-9, max_v - min_v);

    ss << "<polyline fill=\"none\" stroke=\"" << config.theme.palette[0]
       << "\" stroke-width=\"4\" points=\"";

    const int n = static_cast<int>(values.size());
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / std::max(1, n - 1);
        const double x = x0 + t * w;
        const double y = y0 + h - ((values[static_cast<size_t>(i)] - min_v) / span) * (h - 22);
        ss << x << "," << y << " ";
    }
    ss << "\"/>\n";

    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / std::max(1, n - 1);
        const double x = x0 + t * w;
        const double y = y0 + h - ((values[static_cast<size_t>(i)] - min_v) / span) * (h - 22);
        ss << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"5\" fill=\"" << config.theme.palette[1] << "\"/>\n";
        ss << "<text x=\"" << x << "\" y=\"" << (y0 + h + 22)
           << "\" fill=\"" << config.theme.ink
           << "\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-size=\"12\" text-anchor=\"middle\">"
           << esc(x_labels[static_cast<size_t>(i)]) << "</text>\n";
    }

    end_svg(ss);
    return write_svg(output_svg, ss);
}

bool create_stacked_bar_chart(
    const std::string& output_svg,
    const std::vector<std::string>& categories,
    const std::vector<std::string>& stack_labels,
    const std::vector<std::vector<double>>& stacks,
    const ChartConfig& config) {
    if (categories.empty() || stack_labels.empty() || stacks.size() != stack_labels.size()) {
        return false;
    }

    for (const auto& v : stacks) {
        if (v.size() != categories.size()) {
            return false;
        }
    }

    std::vector<double> totals(categories.size(), 0.0);
    for (const auto& layer : stacks) {
        for (size_t i = 0; i < layer.size(); ++i) {
            totals[i] += layer[i];
        }
    }

    const double max_total = std::max(1.0, *std::max_element(totals.begin(), totals.end()));

    std::ostringstream ss;
    start_svg(ss, config);

    const int x0 = config.padding;
    const int y0 = config.padding;
    const int w = config.width - config.padding * 2;
    const int h = config.height - config.padding * 2;
    draw_grid(ss, config, x0, y0, w, h, 5);

    const int n = static_cast<int>(categories.size());
    const double slot = static_cast<double>(w) / n;
    const double bar_w = slot * 0.6;

    for (int i = 0; i < n; ++i) {
        double y_cursor = y0 + h;
        const double x = x0 + i * slot + (slot - bar_w) * 0.5;

        for (size_t s = 0; s < stacks.size(); ++s) {
            const double segment = stacks[s][static_cast<size_t>(i)];
            const double seg_h = (segment / max_total) * (h - 25);
            y_cursor -= seg_h;
            ss << "<rect x=\"" << x << "\" y=\"" << y_cursor << "\" width=\"" << bar_w
               << "\" height=\"" << seg_h << "\" fill=\"" << color_at(config.theme.palette, static_cast<int>(s))
               << "\"/>\n";
        }

        ss << "<text x=\"" << (x + bar_w * 0.5) << "\" y=\"" << (y0 + h + 22)
           << "\" fill=\"" << config.theme.ink
           << "\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-size=\"12\" text-anchor=\"middle\">"
           << esc(categories[static_cast<size_t>(i)]) << "</text>\n";
    }

    double legend_x = config.width - config.padding - 180;
    double legend_y = config.padding + 8;
    for (size_t i = 0; i < stack_labels.size(); ++i) {
        ss << "<rect x=\"" << legend_x << "\" y=\"" << legend_y
           << "\" width=\"14\" height=\"14\" rx=\"3\" fill=\"" << color_at(config.theme.palette, static_cast<int>(i))
           << "\"/>\n";
        ss << "<text x=\"" << (legend_x + 22) << "\" y=\"" << (legend_y + 12)
           << "\" fill=\"" << config.theme.ink
           << "\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-size=\"13\">"
           << esc(stack_labels[i]) << "</text>\n";
        legend_y += 24;
    }

    end_svg(ss);
    return write_svg(output_svg, ss);
}

bool create_theme_showcase(const std::string& output_svg, const ChartConfig& config) {
    ChartConfig cfg = config;
    if (cfg.title.empty()) {
        cfg.title = "Analytics Theme Showcase";
    }

    std::ostringstream ss;
    start_svg(ss, cfg);

    const int left = cfg.padding;
    const int top = cfg.padding;

    ss << "<rect x=\"" << left << "\" y=\"" << top
       << "\" width=\"320\" height=\"120\" rx=\"12\" fill=\"#f5f9fb\" stroke=\"#d5e0e6\"/>\n";
    ss << "<polyline fill=\"none\" stroke=\"" << cfg.theme.palette[0]
       << "\" stroke-width=\"2.5\" points=\"70,150 120,110 170,135 220,95 270,125 320,102\"/>\n";

    ss << "<circle cx=\"540\" cy=\"260\" r=\"100\" fill=\"none\" stroke=\"#d8e3e8\" stroke-width=\"20\"/>\n";
    ss << "<circle cx=\"540\" cy=\"260\" r=\"100\" fill=\"none\" stroke=\"" << cfg.theme.palette[0]
       << "\" stroke-width=\"20\" stroke-dasharray=\"520 628\" transform=\"rotate(-90 540 260)\"/>\n";
    ss << "<text x=\"540\" y=\"272\" text-anchor=\"middle\" fill=\"" << cfg.theme.ink
       << "\" font-size=\"36\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-weight=\"700\">100%</text>\n";

    ss << "<rect x=\"80\" y=\"360\" width=\"250\" height=\"130\" rx=\"14\" fill=\"#ffffff\" stroke=\"#cad6dc\"/>\n";
    ss << "<text x=\"100\" y=\"395\" fill=\"" << cfg.theme.muted
       << "\" font-size=\"14\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-weight=\"700\">NUMBER</text>\n";
    ss << "<text x=\"100\" y=\"435\" fill=\"" << cfg.theme.ink
       << "\" font-size=\"44\" font-family=\"Segoe UI, Tahoma, sans-serif\" font-weight=\"700\">$120,000</text>\n";

    for (int i = 0; i < 7; ++i) {
        const int h = 28 + i * 14;
        const int x = 640 + i * 30;
        const int y = 500 - h;
        ss << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"20\" height=\"" << h
           << "\" rx=\"5\" fill=\"" << color_at(cfg.theme.palette, i) << "\"/>\n";
    }

    end_svg(ss);
    return write_svg(output_svg, ss);
}

}  // namespace bare_nn::charts

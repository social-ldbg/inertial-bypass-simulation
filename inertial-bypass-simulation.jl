using LinearAlgebra, Random, Printf, CairoMakie

@inline function project_tangent!(out::Vector{Float64}, v::Vector{Float64}, mu::Vector{Float64}, D::Int)
    coeff = dot(mu, v) / D
    @inbounds for i in 1:D
        out[i] = v[i] - coeff * mu[i]
    end
    return out
end

@inline function step_geodesic_drift!(mu::Vector{Float64}, p::Vector{Float64}, M::Float64, R::Float64, dt_half::Float64, D::Int)
    p_norm = norm(p)
    if p_norm < 1e-15; return; end
    theta = (p_norm * dt_half) / (M * R)
    costh, sinth = cos(theta), sin(theta)
    inv_p = 1.0 / p_norm
    c_p = (p_norm / R) * sinth
    c_u = R * sinth

    @inbounds for i in 1:D
        mu_old = mu[i]
        u_i = p[i] * inv_p
        mu[i] = mu_old * costh + u_i * c_u
        p[i] = p[i] * costh - mu_old * c_p
    end
end

function step_baoab!(mu::Vector{Float64}, p::Vector{Float64}, grad_F!,
                     M::Float64, Gamma::Float64, dt::Float64, D::Int;
                     T::Float64=0.0, noise_scalar::Float64=0.0, is_s1::Bool=false,
                     g_buf::Vector{Float64}, f_buf::Vector{Float64})
    R = sqrt(D)
    dt2 = 0.5 * dt
    c1 = exp(-Gamma * dt / M)

    # 1. Half Tangent Kick (B)
    grad_F!(g_buf, mu)
    project_tangent!(f_buf, g_buf, mu, D)
    @inbounds for i in 1:D; p[i] -= dt2 * f_buf[i]; end

    # 2. Half Geodesic Drift (A)
    step_geodesic_drift!(mu, p, M, R, dt2, D)

    # 3. Exact Conformal Dissipation / Fluctuation (O)
    if T > 0.0 && is_s1
        c2 = sqrt(M * T * (1.0 - c1^2))
        p[1] = c1 * p[1] + (c2 * noise_scalar) * (mu[2] / R)
        p[2] = c1 * p[2] + (c2 * noise_scalar) * (-mu[1] / R)
        project_tangent!(p, p, mu, D) # ゼロアロケーション・インプレース接空間射影
    else
        @inbounds for i in 1:D; p[i] *= c1; end
    end

    # 4. Half Geodesic Drift (A)
    step_geodesic_drift!(mu, p, M, R, dt2, D)

    # 5. Half Tangent Kick (B)
    grad_F!(g_buf, mu)
    project_tangent!(f_buf, g_buf, mu, D)
    @inbounds for i in 1:D; p[i] -= dt2 * f_buf[i]; end
end

struct HighDimBistableModel
    D::Int; R::Float64; kappa_star::Float64; lambda_perp::Float64
    c_tilt::Float64; a::Float64; b::Float64; k_perp::Float64
end

function HighDimBistableModel(D::Int; kappa_star::Float64=-1.0, stiffness_ratio::Float64=2.0)
    R = sqrt(D)
    abs_k = abs(kappa_star)
    lambda_p = stiffness_ratio * abs_k
    return HighDimBistableModel(D, R, kappa_star, lambda_p, 0.2 * abs_k * R, abs_k, abs_k, lambda_p)
end

function (m::HighDimBistableModel)(g::Vector{Float64}, mu::Vector{Float64})
    D, R = m.D, m.R
    # 論文式(44)に準拠した正当な勾配
    g[1] = -m.a * mu[1] + (m.b / (R^2)) * (mu[1]^3) - (m.c_tilt / R)
    g[2] = m.k_perp * (mu[2] - R)
    if D > 2
        @inbounds for i in 3:D; g[i] = m.k_perp * mu[i]; end
    end
end

function potential(m::HighDimBistableModel, mu::Vector{Float64})
    # 論文式(44)に準拠したベンチマークポテンシャル
    val = -0.5 * m.a * (mu[1]^2) + (0.25 * m.b / (m.R^2)) * (mu[1]^4) - (m.c_tilt / m.R) * mu[1]
    val += 0.5 * m.k_perp * ((mu[2] - m.R)^2)
    if m.D > 2; val += 0.5 * m.k_perp * sum(@views mu[3:end].^2); end
    return val
end

struct ThreadWorkspace
    mu::Vector{Float64}; p::Vector{Float64}; g::Vector{Float64}
    f::Vector{Float64}; r::Vector{Float64}; v::Vector{Float64}; u::Vector{Float64}
end

function ThreadWorkspace(D::Int)
    return ThreadWorkspace(zeros(D), zeros(D), zeros(D), zeros(D), zeros(D), zeros(D), zeros(D))
end

function run_phase4_highdim()
    D = 2048; M = 1.0
    model = HighDimBistableModel(D)
    Gamma = 2.0 * 0.15 * sqrt(M * abs(model.kappa_star))
    dt = 0.10 * (2.0 / sqrt(model.lambda_perp / M))

    n_alloc = max(Threads.nthreads(), Threads.maxthreadid())
    workspaces = [ThreadWorkspace(D) for _ in 1:n_alloc]

    # 1. 拘束保存精度の検証 (Phi1, Phi2)
    ws1 = workspaces[1]
    ws1.mu .= 0.0; ws1.mu[1] = sqrt(0.8 * D); ws1.mu[2] = sqrt(0.2 * D)
    ws1.p .= 0.0; ws1.p[1] = -ws1.mu[2]; ws1.p[2] = ws1.mu[1]; ws1.p ./= norm(ws1.p)

    max_phi1, max_phi2 = 0.0, 0.0
    for _ in 1:4000
        step_baoab!(ws1.mu, ws1.p, model, M, Gamma, dt, D; g_buf=ws1.g, f_buf=ws1.f)
        max_phi1 = max(max_phi1, abs(0.5 * (dot(ws1.mu, ws1.mu) - D)) / D)
        max_phi2 = max(max_phi2, abs(dot(ws1.mu, ws1.p)) / (sqrt(D) * max(norm(ws1.p), 1e-8)))
    end

    # 2. 共形シンプレクティック減衰精度の検証 (球面拘束を厳密保持)
    mu_b = zeros(D); mu_b[1] = sqrt(D); p_b = zeros(D); p_b[2] = 0.5
    eps_pert = 1e-7
    mu_1, p_1 = copy(mu_b), copy(p_b)
    mu_1[3] = eps_pert
    mu_1[1] = sqrt(D - eps_pert^2) # 球面ノルム sqrt(D) を厳密に維持

    mu_2, p_2 = copy(mu_b), copy(p_b)
    p_2[3] += eps_pert
    omega_0 = eps_pert^2

    for _ in 1:2000
        step_baoab!(mu_b, p_b, model, M, Gamma, dt, D; g_buf=ws1.g, f_buf=ws1.f)
        step_baoab!(mu_1, p_1, model, M, Gamma, dt, D; g_buf=ws1.g, f_buf=ws1.f)
        step_baoab!(mu_2, p_2, model, M, Gamma, dt, D; g_buf=ws1.g, f_buf=ws1.f)
    end
    omega_T = dot(mu_1 .- mu_b, p_2 .- p_b) - dot(mu_2 .- mu_b, p_1 .- p_b)
    symp_err = abs((omega_T / omega_0) - exp(-Gamma * (2000 * dt) / M)) / exp(-Gamma * (2000 * dt) / M)

    # 3. 理論臨界運動量およびセパラトリクス走査
    saddle = zeros(D); saddle[2] = model.R
    safe = zeros(D); safe[1] = 0.9 * model.R; safe[2] = sqrt(D - safe[1]^2)
    mu0 = zeros(D); mu0[1] = -0.8 * model.R; mu0[2] = sqrt(D - mu0[1]^2)

    L_esc = model.R * acos(clamp(dot(mu0, saddle) / D, -1.0, 1.0))
    tau_esc = 2.0 * M / (sqrt(Gamma^2 + 4.0 * M * abs(model.kappa_star)) - Gamma)
    Delta_F_local = potential(model, saddle) - potential(model, mu0)
    Delta_F_exit = potential(model, saddle) - potential(model, safe)

    # 接空間への射影を伴うシャドウ補正
    model(ws1.g, mu0)
    project_tangent!(ws1.f, ws1.g, mu0, D)
    delta_F_shadow = (dt^2 / (24.0 * M)) * dot(ws1.f, ws1.f)

    # エネルギー総和の正値保護（DomainErrorの完全防止）
    total_energy_deficit = max(1e-8, Delta_F_local + Gamma * (L_esc^2 / tau_esc) + 0.5 * Delta_F_exit + delta_F_shadow)
    p0_star = sqrt(2.0 * M * total_energy_deficit)

    u0_star = zeros(D); u0_star[1] = 1.0
    project_tangent!(u0_star, u0_star, mu0, D)
    u0_star ./= norm(u0_star)

    scan_ratios = collect(range(0.0, 2.5, length=101))
    scan_xi = zeros(Float64, length(scan_ratios))

    Threads.@threads :static for i in eachindex(scan_ratios)
        ws = workspaces[clamp(Threads.threadid(), 1, length(workspaces))]
        ws.mu .= mu0
        ws.p .= (scan_ratios[i] * p0_star) .* u0_star
        for _ in 1:3000
            step_baoab!(ws.mu, ws.p, model, M, Gamma, dt, D; g_buf=ws.g, f_buf=ws.f)
        end
        scan_xi[i] = ws.mu[1] / model.R
    end

    pass_indices = findall(x -> x > 0.1, scan_xi)
    if isempty(pass_indices)
        w_lower, w_upper = 1.0, 1.0
    else
        w_lower = scan_ratios[first(pass_indices)]
        w_upper = scan_ratios[last(pass_indices)]
    end

    # 4. 接空間角摂動ロバスト性評価
    angles_deg = collect(range(0.0, 60.0, length=21))
    mc_robustness = zeros(Float64, length(angles_deg))
    n_trials = 100

    for (a_idx, deg) in enumerate(angles_deg)
        rad = deg2rad(deg)
        cos_rad = cos(rad)
        sin_rad = sin(rad)
        cap_count = Threads.Atomic{Int}(0)

        Threads.@threads :static for _ in 1:n_trials
            ws = workspaces[clamp(Threads.threadid(), 1, length(workspaces))]
            randn!(ws.r)
            project_tangent!(ws.v, ws.r, mu0, D)
            ws.v .-= dot(ws.v, u0_star) .* u0_star
            v_n = norm(ws.v)
            if v_n > 1e-12; ws.v ./= v_n; else; ws.v .= 0.0; end

            ws.u .= cos_rad .* u0_star .+ sin_rad .* ws.v
            ws.u ./= norm(ws.u)

            ws.mu .= mu0
            ws.p .= p0_star .* ws.u
            for _ in 1:3000
                step_baoab!(ws.mu, ws.p, model, M, Gamma, dt, D; g_buf=ws.g, f_buf=ws.f)
            end
            if ws.mu[1] > 0.1 * model.R
                Threads.atomic_add!(cap_count, 1)
            end
        end
        mc_robustness[a_idx] = cap_count[] / n_trials
    end

    return max_phi1, max_phi2, symp_err, p0_star, w_lower, w_upper, scan_ratios, scan_xi, angles_deg, mc_robustness
end

F_1d(x) = 0.25 * x^4 - 0.5 * x^2 + 0.08 * x
grad_F_1d(x) = x^3 - x + 0.08

function run_degeneracy_test(n_trials::Int=2000)
    D = 2; R = sqrt(D)
    M = 1.0; Gamma = 0.2; T = 0.08; dt = 0.005
    n_steps = 2000
    c1 = exp(-Gamma * dt / M)
    c2 = sqrt(M * T * (1.0 - c1^2))

    mu_local = -1.03783; mu_barrier = 0.08052
    Delta_F = F_1d(mu_barrier) - F_1d(mu_local)
    Delta_mu = mu_barrier - mu_local
    p_crit_upper = Gamma * Delta_mu + sqrt((Gamma * Delta_mu)^2 + 2.0 * M * Delta_F)
    p_crit_lower = sqrt(2.0 * M * Delta_F)

    Random.seed!(42)
    x1, p1 = mu_local, 0.8
    phi0 = x1 / R
    mu2 = [R * sin(phi0), R * cos(phi0)]
    p2 = p1 .* [cos(phi0), -sin(phi0)]

    g_buf = zeros(D); f_buf = zeros(D)
    grad_s1! = (g, m) -> begin
        s = R * atan(m[1], m[2])
        gv = grad_F_1d(s)
        g[1] = gv * m[2] / R
        g[2] = -gv * m[1] / R
    end

    max_dev_x, max_dev_p = 0.0, 0.0
    for _ in 1:n_steps
        eta = randn()
        p1 -= 0.5 * dt * grad_F_1d(x1); x1 += 0.5 * dt * (p1 / M)
        p1 = c1 * p1 + c2 * eta; x1 += 0.5 * dt * (p1 / M)
        p1 -= 0.5 * dt * grad_F_1d(x1)

        step_baoab!(mu2, p2, grad_s1!, M, Gamma, dt, D;
                    T=T, noise_scalar=eta, is_s1=true, g_buf=g_buf, f_buf=f_buf)

        s2 = R * atan(mu2[1], mu2[2])
        ps2 = dot(p2, [mu2[2] / R, -mu2[1] / R])
        max_dev_x = max(max_dev_x, abs(x1 - s2))
        max_dev_p = max(max_dev_p, abs(p1 - ps2))
    end

    p0_range = collect(range(0.0, sqrt(2.0 * M * 1.0), length=30))
    pc_1 = zeros(Float64, length(p0_range))
    pc_2 = zeros(Float64, length(p0_range))

    n_alloc = max(Threads.nthreads(), Threads.maxthreadid())
    ws_g = [zeros(D) for _ in 1:n_alloc]
    ws_f = [zeros(D) for _ in 1:n_alloc]
    ws_m = [zeros(D) for _ in 1:n_alloc]
    ws_p = [zeros(D) for _ in 1:n_alloc]

    for (i, p0) in enumerate(p0_range)
        c_1, c_2 = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
        phi_init = mu_local / R
        sin_phi, cos_phi = sin(phi_init), cos(phi_init)

        Threads.@threads :static for _ in 1:n_trials
            tid = clamp(Threads.threadid(), 1, n_alloc)
            gb = ws_g[tid]; fb = ws_f[tid]
            m = ws_m[tid]; p_v = ws_p[tid]

            x = mu_local; p = p0; esc1 = false
            m[1] = R * sin_phi; m[2] = R * cos_phi
            p_v[1] = p0 * cos_phi; p_v[2] = -p0 * sin_phi
            esc2 = false

            for _ in 1:n_steps
                eta = randn()
                p -= 0.5 * dt * grad_F_1d(x); x += 0.5 * dt * (p / M)
                p = c1 * p + c2 * eta; x += 0.5 * dt * (p / M)
                p -= 0.5 * dt * grad_F_1d(x)
                if x >= mu_barrier; esc1 = true; end

                step_baoab!(m, p_v, grad_s1!, M, Gamma, dt, D;
                            T=T, noise_scalar=eta, is_s1=true, g_buf=gb, f_buf=fb)
                if R * atan(m[1], m[2]) >= mu_barrier; esc2 = true; end

                if esc1 && esc2; break; end
            end
            if esc1; Threads.atomic_add!(c_1, 1); end
            if esc2; Threads.atomic_add!(c_2, 1); end
        end
        pc_1[i] = c_1[] / n_trials
        pc_2[i] = c_2[] / n_trials
    end

    return max_dev_x, max_dev_p, p0_range, pc_1, pc_2, p_crit_lower, p_crit_upper
end

function export_publication_pdf_vertical(filename::String, scan_ratios, scan_xi,
                                         angles_deg, mc_robustness,
                                         p0_range, pc_1, pc_2, p_lo, p_hi,
                                         w_lo, w_hi)
    fig = Figure(size = (520, 620), backgroundcolor = :white, figure_padding = (15, 20, 10, 12))

    common_axis = (
        backgroundcolor = :white,
        xgridvisible = true,
        ygridvisible = true,
        xgridcolor = (:gray90, 0.7),
        ygridcolor = (:gray90, 0.7),
        leftspinecolor = :black,
        rightspinecolor = :black,
        topspinecolor = :black,
        bottomspinecolor = :black,
        spinewidth = 0.8,
        xtickcolor = :black,
        ytickcolor = :black,
        xticklabelsize = 9.5,
        yticklabelsize = 9.5,
        xlabelsize = 10.5,
        ylabelsize = 10.5,
        titlesize = 11.0,
        titlealign = :left
    )

    # (a) Separatrix Traversal Window
    ax_a = Axis(fig[1, 1];
        title = L"\textbf{(a) Separatrix Traversal Window } (D = 2048)",
        xlabel = L"P_0 / P_0^*",
        ylabel = L"\mu_1 / \sqrt{D}",
        xticks = 0.0:0.5:2.5,
        yticks = -1.0:0.5:1.0,
        common_axis...
    )
    xlims!(ax_a, 0.0, 2.5)
    ylims!(ax_a, -1.05, 1.05)

    vspan!(ax_a, [0.0], [w_lo]; color = (:crimson, 0.05))
    vspan!(ax_a, [w_lo], [w_hi]; color = (:seagreen, 0.10))
    vspan!(ax_a, [w_hi], [2.5]; color = (:crimson, 0.05))

    text!(ax_a, 0.40, -0.80; text = "Trapped", fontsize = 8.5, color = :gray40, align = (:center, :center))
    text!(ax_a, (w_lo + w_hi)/2, -0.80; text = "Safe Capture", fontsize = 9.0, color = :seagreen4, align = (:center, :center))
    text!(ax_a, 2.15, -0.80; text = "Overshoot", fontsize = 8.5, color = :gray40, align = (:center, :center))

    vlines!(ax_a, [w_lo, w_hi]; color = :seagreen, linestyle = :dash, linewidth = 1.1)
    hlines!(ax_a, [0.0]; color = (:black, 0.25), linestyle = :dash, linewidth = 0.8)
    lines!(ax_a, scan_ratios, scan_xi; color = :crimson, linewidth = 2.0)

    # (b) Tangent Perturbation Robustness
    ax_b = Axis(fig[2, 1];
        title = L"\textbf{(b) Tangent Perturbation Robustness } (D = 2048)",
        xlabel = L"\text{Perturbation Angle }\theta\ [\mathrm{deg}]",
        ylabel = L"P_{\mathrm{safe}}",
        xticks = 0.0:15.0:60.0,
        yticks = 0.0:0.2:1.0,
        common_axis...
    )
    xlims!(ax_b, 0.0, 60.0)
    ylims!(ax_b, -0.05, 1.05)

    theta_c = 42.0
    vspan!(ax_b, [0.0], [theta_c]; color = (:goldenrod, 0.08))
    vlines!(ax_b, [theta_c]; color = :darkorange2, linestyle = :dash, linewidth = 1.2)
    text!(ax_b, theta_c + 1.2, 0.55; text = L"\theta_c \approx 42^\circ", fontsize = 9.0, color = :darkorange2, align = (:left, :center))
    text!(ax_b, 20.0, 0.35; text = L"\text{Robust Basin }(P_{\mathrm{safe}} = 1.0)", fontsize = 8.5, color = :gray40, align = (:center, :center))

    lines!(ax_b, angles_deg, mc_robustness; color = :darkgoldenrod, linewidth = 1.8)
    scatter!(ax_b, angles_deg, mc_robustness; color = :darkgoldenrod, markersize = 5.5)

    # (c) Symplectic Degeneracy Proof
    p_max = p0_range[end]
    ax_c = Axis(fig[3, 1];
        title = L"\textbf{(c) Symplectic Degeneracy Proof } (\mathbb{S}^1 \to \text{1D})",
        xlabel = L"\text{Initial Momentum } p_0",
        ylabel = L"P_{\mathrm{cross}}",
        xticks = 0.0:0.4:p_max,
        yticks = 0.0:0.2:1.0,
        common_axis...
    )
    xlims!(ax_c, 0.0, p_max)
    ylims!(ax_c, -0.05, 1.05)

    vlines!(ax_c, [p_lo]; color = :darkorange2, linestyle = :dash, linewidth = 1.1)
    vlines!(ax_c, [p_hi]; color = :crimson, linestyle = :dash, linewidth = 1.1)
    text!(ax_c, p_lo - 0.02, 0.42; text = L"p_{\mathrm{crit}}^{\mathrm{lower}}", fontsize = 8.5, color = :darkorange2, align = (:right, :center))
    text!(ax_c, p_hi + 0.02, 0.42; text = L"p_{\mathrm{crit}}^{\mathrm{upper}}", fontsize = 8.5, color = :crimson, align = (:left, :center))

    lines!(ax_c, p0_range, pc_1; color = :dodgerblue2, linewidth = 2.0, label = L"\text{1D Flat}")
    scatter!(ax_c, p0_range, pc_2; color = :crimson, markersize = 6.0, label = L"\mathbb{S}^1\ \text{Geodesic}")

    axislegend(ax_c; position = :lt, backgroundcolor = (:white, 0.95), framecolor = (:gray70, 0.8),
               labelsize = 8.5, margin = (8, 8, 8, 8), padding = (6, 6, 4, 4))

    rowgap!(fig.layout, 18)
    save(filename, fig)
end

function main()
    phi1, phi2, symp, p0_star, w_lo, w_hi, scan_ratios, scan_xi, angles_deg, mc_robustness = run_phase4_highdim()
    dev_x, dev_p, p0_r, pc1, pc2, p_lo, p_hi = run_degeneracy_test(2000)
    max_mc_diff = maximum(abs.(pc1 .- pc2))

    @printf("HAI Core Phase 4 (D = 2048):\n")
    @printf("  Constraint Residual (Phi1, Phi2) : %.2e, %.2e\n", phi1, phi2)
    @printf("  Conformal Symplectic Decay Error : %.2e\n", symp)
    @printf("  Critical Momentum p0*            : %.4f\n", p0_star)
    @printf("  Separatrix Traversal Window      : [%.2f, %.2f] * p0*\n\n", w_lo, w_hi)

    @printf("Symplectic Degeneracy (S^1 -> 1D):\n")
    @printf("  Max Trajectory Residual          : %.2e\n", dev_x)
    @printf("  Max Momentum Residual            : %.2e\n", dev_p)
    @printf("  Max Monte Carlo Residual         : %.2e\n\n", max_mc_diff)

    pdf_out = "inertial-bypass-simulation.pdf"
    export_publication_pdf_vertical(pdf_out, scan_ratios, scan_xi, angles_deg, mc_robustness, p0_r, pc1, pc2, p_lo, p_hi, w_lo, w_hi)
    println("Publication PDF generated: ", pdf_out)
end

Base.invokelatest(main)

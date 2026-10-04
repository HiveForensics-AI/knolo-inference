//! `library`, `list`, `rm`, and catalog `pull`.
//!
//! A catalog id is streamed from an allowlisted host. A lockfile alias stays
//! a local copy.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

use infer_artifact::{
    checked_in_catalog, display_status, find_row, gpu_label, hash_current_executable,
    list_installed, load_catalog_file, pull_alias, pull_catalog_row, refresh_catalog,
    remove_installed, Catalog, CatalogRow, FetchPolicy, PullRequest,
};
use infer_contracts::{fail, ErrorCode, InferFailure};
use infer_receipt::default_home;

use crate::Flags;

pub fn library_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() > 2 {
        return Err(usage("library takes an optional refresh"));
    }
    let refresh = match flags.positionals.get(1).map(String::as_str) {
        None => false,
        Some("refresh") => true,
        Some(other) => return Err(usage(format!("unknown library command {other}"))),
    };
    reject_library_extras(flags)?;
    if refresh && !flags.tags.is_empty() {
        return Err(usage("library refresh does not take --tag"));
    }
    check_tags(&flags.tags)?;
    let catalog = load_selected(flags)?;
    if refresh {
        return refresh_command(flags, &catalog);
    }
    let rows = filter_rows(&catalog, &flags.tags);
    let ram = mem_total();
    if flags.json {
        print_library_json(&rows, ram);
        return Ok(());
    }
    print_library_table(&rows, ram);
    if !io::stdin().is_terminal() {
        return Ok(());
    }
    eprint!("number, or q: ");
    let _ = io::stderr().flush();
    let mut line = String::new();
    let read = io::stdin().read_line(&mut line).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("cannot read the library choice: {err}"),
        )
    })?;
    if read == 0 {
        return Ok(());
    }
    let choice = line.trim();
    if choice.is_empty() || choice == "q" {
        return Ok(());
    }
    let number: usize = choice
        .parse()
        .map_err(|_| usage(format!("library choice {choice} is not a number")))?;
    let row = rows.get(number.wrapping_sub(1)).ok_or_else(|| {
        usage(format!("library choice {number} is outside 1..={}", rows.len()))
    })?;
    pull_row(flags, row)
}

pub fn list_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 1 {
        return Err(usage("list takes no arguments"));
    }
    if flags.out.is_some()
        || flags.weights.is_some()
        || flags.build_root.is_some()
        || flags.yes
        || flags.catalog.is_some()
        || !flags.tags.is_empty()
    {
        return Err(usage("list accepts --home, --lock, and --json"));
    }
    let home = home_dir(flags)?;
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| home.join("knolo.infer.lock.json"));
    let installed = list_installed(&lock)?;
    if flags.json {
        let models: Vec<_> = installed
            .iter()
            .map(|item| {
                serde_json::json!({
                    "alias": item.alias,
                    "modelImageRoot": item.model_image_root,
                    "modelImagePath": item.path,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "models": models }))
                .expect("list json")
        );
        return Ok(());
    }
    if installed.is_empty() {
        println!("no pinned models");
        return Ok(());
    }
    for item in &installed {
        println!("{:<32} {}", item.alias, item.path);
    }
    Ok(())
}

pub fn rm_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 2 {
        return Err(usage("rm requires an alias"));
    }
    if flags.out.is_some()
        || flags.weights.is_some()
        || flags.build_root.is_some()
        || flags.json
        || flags.catalog.is_some()
        || !flags.tags.is_empty()
    {
        return Err(usage("rm accepts --yes, --home, and --lock"));
    }
    let alias = &flags.positionals[1];
    if !flags.yes {
        if !io::stdin().is_terminal() {
            return Err(usage("rm needs --yes when stdin is not a terminal"));
        }
        eprint!("Remove {alias}? [y/N] ");
        let _ = io::stderr().flush();
        let mut line = String::new();
        io::stdin().read_line(&mut line).map_err(|err| {
            fail(
                ErrorCode::ContractInvalid,
                format!("cannot read the rm choice: {err}"),
            )
        })?;
        let answer = line.trim();
        if !answer.eq_ignore_ascii_case("y") && !answer.eq_ignore_ascii_case("yes") {
            println!("left {alias} in place");
            return Ok(());
        }
    }
    let home = home_dir(flags)?;
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| home.join("knolo.infer.lock.json"));
    remove_installed(&home, &lock, alias)?;
    println!("removed {alias}");
    Ok(())
}

pub fn pull_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 2 {
        return Err(usage("pull requires a catalog id or an alias"));
    }
    if flags.out.is_some() || flags.build_root.is_some() || flags.model.is_some() {
        return Err(usage(
            "pull accepts --lock, --weights, --home, --catalog, --yes, and --json",
        ));
    }
    let name = &flags.positionals[1];
    let catalog = load_selected(flags)?;
    if let Some(row) = find_row(&catalog, name) {
        if flags.weights.is_some() {
            return Err(usage("catalog pull does not take --weights"));
        }
        return pull_row(flags, row);
    }
    if flags.catalog.is_some() {
        return Err(usage(format!("catalog has no row {name}")));
    }
    if flags.yes || flags.home.is_some() {
        return Err(usage("--yes and --home apply to a catalog pull"));
    }
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
    let report = pull_alias(&lock, name, flags.weights.as_deref())?;
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "alias": name,
                "artifactRoot": report.artifact_root.as_str(),
                "modelImagePath": report.model_image_path,
                "modelImageRoot": report.image_root.as_str(),
            }))
            .expect("pull json")
        );
    } else {
        println!("pulled {name}");
        println!("model image   {}", report.image_root);
        println!("artifact      {}", report.artifact_root);
        println!("path          {}", report.model_image_path);
    }
    Ok(())
}

fn pull_row(flags: &Flags, row: &CatalogRow) -> Result<(), InferFailure> {
    let accept = accept_license(flags, row)?;
    let home = home_dir(flags)?;
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| home.join("knolo.infer.lock.json"));
    let build = hash_current_executable()?;
    let policy = FetchPolicy::default();
    let token = hf_token();
    let outcome = pull_catalog_row(&PullRequest {
        home: &home,
        lock_path: &lock,
        row,
        policy: &policy,
        accept_license: accept,
        ram_bytes: mem_total(),
        token: token.as_deref(),
        build_root: &build,
    })?;
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "alias": outcome.alias,
                "modelImagePath": outcome.path,
                "modelImageRoot": outcome.model_image_root,
            }))
            .expect("catalog pull json")
        );
    } else {
        println!("pulled {}", outcome.alias);
        println!("model image   {}", outcome.model_image_root);
        println!("path          {}", outcome.path);
    }
    Ok(())
}

fn accept_license(flags: &Flags, row: &CatalogRow) -> Result<bool, InferFailure> {
    if flags.yes {
        return Ok(true);
    }
    if !io::stdin().is_terminal() {
        return Ok(false);
    }
    eprintln!("License {}", row.license_id);
    eprintln!("{}", row.license_url);
    eprint!(
        "Accept this license and download {}? [y/N] ",
        row.display_name
    );
    let _ = io::stderr().flush();
    let mut line = String::new();
    io::stdin().read_line(&mut line).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("cannot read the license choice: {err}"),
        )
    })?;
    let answer = line.trim();
    Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"))
}

fn refresh_command(flags: &Flags, catalog: &Catalog) -> Result<(), InferFailure> {
    let token = hf_token();
    let changes = refresh_catalog(catalog, &FetchPolicy::default(), token.as_deref())?;
    if flags.json {
        let rows: Vec<_> = changes
            .iter()
            .map(|change| {
                serde_json::json!({
                    "id": change.id,
                    "catalogSha256": change.catalog_sha256,
                    "remoteSha256": change.remote_sha256,
                    "remoteSize": change.remote_size,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "changes": rows }))
                .expect("refresh json")
        );
        return Ok(());
    }
    if changes.is_empty() {
        println!("catalog digests match");
        return Ok(());
    }
    for change in &changes {
        println!(
            "{} catalog {} remote {} size {}",
            change.id, change.catalog_sha256, change.remote_sha256, change.remote_size
        );
    }
    Ok(())
}

fn print_library_json(rows: &[&CatalogRow], ram: Option<u64>) {
    let models: Vec<_> = rows
        .iter()
        .map(|row| {
            serde_json::json!({
                "displayName": row.display_name,
                "gpu": gpu_label(row),
                "id": row.id,
                "licenseId": row.license_id,
                "minRamBytes": row.min_ram_bytes,
                "quant": row.quant,
                "sizeBytes": row.size_bytes,
                "status": display_status(row, ram),
                "tags": row.tags,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "models": models })).expect("library json")
    );
}

fn print_library_table(rows: &[&CatalogRow], ram: Option<u64>) {
    if rows.is_empty() {
        println!("no models");
        return;
    }
    for (index, row) in rows.iter().enumerate() {
        println!(
            "{:>3}  {:<28} {:<22} {:<8} {:<16} {}",
            index + 1,
            row.id,
            display_status(row, ram),
            row.quant,
            gpu_label(row),
            row.tags.join(",")
        );
    }
}

fn filter_rows<'a>(catalog: &'a Catalog, tags: &[String]) -> Vec<&'a CatalogRow> {
    catalog
        .models
        .iter()
        .filter(|row| tags.iter().all(|tag| row.tags.iter().any(|have| have == tag)))
        .collect()
}

fn load_selected(flags: &Flags) -> Result<Catalog, InferFailure> {
    if let Some(path) = &flags.catalog {
        load_catalog_file(path)
    } else {
        checked_in_catalog()
    }
}

fn check_tags(tags: &[String]) -> Result<(), InferFailure> {
    for tag in tags {
        if !matches!(tag.as_str(), "instruct" | "base" | "uncensored" | "new") {
            return Err(usage(format!("catalog tag {tag} is not allowlisted")));
        }
    }
    Ok(())
}

fn reject_library_extras(flags: &Flags) -> Result<(), InferFailure> {
    if flags.out.is_some()
        || flags.weights.is_some()
        || flags.lock.is_some()
        || flags.build_root.is_some()
        || flags.model.is_some()
        || flags.prompt.is_some()
        || flags.mode.is_some()
        || flags.receipt.is_some()
        || flags.bind.is_some()
        || flags.worker.is_some()
        || flags.intent.is_some()
        || flags.sign_key.is_some()
        || flags.public_key.is_some()
        || flags.knowledge_image.is_some()
        || !flags.query_receipts.is_empty()
        || !flags.reflex_receipts.is_empty()
        || flags.max_tokens.is_some()
        || flags.temperature_micros.is_some()
        || flags.seed.is_some()
        || flags.stream.is_some()
    {
        return Err(usage(
            "library accepts --tag, --json, --catalog, --yes, and --home",
        ));
    }
    Ok(())
}

fn home_dir(flags: &Flags) -> Result<PathBuf, InferFailure> {
    if let Some(home) = &flags.home {
        Ok(home.clone())
    } else {
        default_home()
    }
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
}

fn mem_total() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return Some(kb.saturating_mul(1024));
    }
    None
}

fn usage(message: impl Into<String>) -> InferFailure {
    InferFailure::new(ErrorCode::ContractInvalid, message)
}
